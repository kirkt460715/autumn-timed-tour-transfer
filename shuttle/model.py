"""Loading and derived facts for the cross-venue half-day line.

The source tables are the authority.  This module only parses them and derives
facts the contract states explicitly: clock arithmetic, the joint observation
background of a trip, and notice state by effective time.  Nothing here chooses
a plan.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

DATA = Path(__file__).resolve().parents[1] / "data"

# Segments that make up a whole family journey.  ``a`` and ``b`` are venue
# visits on corridors AB and BC; ``ab`` and ``bc`` are the connecting legs.
SEGMENTS = ("a", "ab", "b", "bc")
CORRIDORS = ("AB", "BC")


class SourceError(ValueError):
    """The source tables cannot be read the way the contract says they may."""


def clock(value: str) -> int:
    """Parse HH:MM into minutes after midnight."""
    try:
        hour, minute = (int(part) for part in value.split(":"))
    except (AttributeError, ValueError) as exc:
        raise SourceError(f"不是 HH:MM 时刻: {value!r}") from exc
    if not 0 <= hour < 24 or not 0 <= minute < 60:
        raise SourceError(f"不是 HH:MM 时刻: {value!r}")
    return hour * 60 + minute


def moment(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise SourceError(f"不是 ISO 时刻: {value!r}") from exc


def hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _read(name: str, directory: Path) -> list[dict[str, str]]:
    with (directory / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class Slot:
    slot: str
    venue: str
    service_date: str
    starts: int
    ends: int
    last_checkin: int
    spare: int


@dataclass(frozen=True)
class Ride:
    ride: str
    service_date: str
    corridor: str
    leaves: int
    arrives: int
    capacity: int
    accessible: bool


@dataclass(frozen=True)
class Walk:
    corridor: str
    minutes: int
    allowed_group: str


@dataclass(frozen=True)
class Family:
    family: str
    people: int
    trip: tuple[str, str, str]
    floor: float
    access: str
    moves: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True)
class Reservation:
    reservation: str
    family: str
    ride: str
    seats: int
    state: str
    issued_at: datetime


@dataclass(frozen=True)
class Activation:
    trigger: str
    group: str
    scope: str
    deadline: int
    applies_to: tuple[str, ...]
    facts: str


@dataclass(frozen=True)
class Notice:
    notice: str
    ride: str
    state: str
    effective_at: datetime
    received_at: datetime


@dataclass(frozen=True)
class Observation:
    record: str
    trip: str
    day: str
    corridor: str
    segment: str
    weather: str
    demand: str
    scheduled_minutes: int
    elapsed_minutes: int | None
    outcome: str
    mode: str

    @property
    def deviation(self) -> int:
        """Minutes this segment ran past its schedule; it never ran if absent."""
        if self.elapsed_minutes is None:
            raise SourceError(f"{self.record} 没有用时，不能计算偏差")
        return self.elapsed_minutes - self.scheduled_minutes


@dataclass(frozen=True)
class TripObservation:
    """All segments recorded for one historical trip."""

    trip: str
    day: str
    weather: str
    demand: str
    segments: Mapping[str, Observation]

    def segment(self, name: str) -> Observation | None:
        return self.segments.get(name)

    @property
    def observed(self) -> bool:
        """Every recorded segment ran to completion on that day."""
        return bool(self.segments) and all(
            item.outcome == "observed" and item.elapsed_minutes is not None
            for item in self.segments.values()
        )

    def state(self, name: str) -> str:
        """``observed`` / ``cancelled`` / ``incomplete`` / ``missing``."""
        item = self.segments.get(name)
        if item is None:
            return "missing"
        if item.outcome != "observed" or item.elapsed_minutes is None:
            return "incomplete" if item.outcome == "incomplete" else item.outcome
        return "observed"


@dataclass
class SourceData:
    slots: dict[str, Slot]
    rides: dict[str, Ride]
    walks: dict[tuple[str, str], Walk]
    families: dict[str, Family]
    reservations: dict[str, Reservation]
    activations: dict[str, Activation]
    notices: dict[str, Notice]
    trips: dict[str, TripObservation]

    def walk_minutes(self, corridor: str, access: str) -> int | None:
        walk = self.walks.get((corridor, access))
        return None if walk is None else walk.minutes

    def corridor_rides(self, corridor: str) -> tuple[Ride, ...]:
        return tuple(r for r in self.rides.values() if r.corridor == corridor)

    def locked_reservations(self, ride: str | None = None) -> tuple[Reservation, ...]:
        return tuple(
            r
            for r in self.reservations.values()
            if r.state == "locked" and (ride is None or r.ride == ride)
        )

    def trips_with(self, *segments: str) -> tuple[TripObservation, ...]:
        return tuple(t for t in self.trips.values() if all(t.state(s) != "missing" for s in segments))

    def shared_days(self, corridor_a: str, corridor_b: str) -> set[str]:
        """Days on which both corridors carry at least one record."""
        out: set[str] = set()
        for trip in self.trips.values():
            seen = {obs.corridor for obs in trip.segments.values()}
            if corridor_a in seen and corridor_b in seen:
                out.add(trip.day)
        return out


def load(directory: Path | str | None = None) -> SourceData:
    directory = Path(directory) if directory is not None else DATA
    slots = {
        row["slot"]: Slot(
            row["slot"],
            row["venue"],
            row["service_date"],
            clock(row["starts"]),
            clock(row["ends"]),
            clock(row["last_checkin"]),
            int(row["spare"]),
        )
        for row in _read("slots", directory)
    }
    rides = {
        row["ride"]: Ride(
            row["ride"],
            row["service_date"],
            row["corridor"],
            clock(row["leaves"]),
            clock(row["arrives"]),
            int(row["capacity"]),
            row["accessible"] == "yes",
        )
        for row in _read("rides", directory)
    }
    walks = {
        (row["corridor"], row["allowed_group"]): Walk(
            row["corridor"], int(row["minutes"]), row["allowed_group"]
        )
        for row in _read("walks", directory)
    }
    families = {
        row["family"]: Family(
            row["family"],
            int(row["people"]),
            tuple(row["trip"].split("|")),  # type: ignore[arg-type]
            float(row["floor"]),
            row["access"],
            (
                tuple(row["move_first"].split("|")),
                tuple(row["move_second"].split("|")),
                tuple(row["move_third"].split("|")),
            ),
        )
        for row in _read("family_route", directory)
    }
    reservations = {
        row["reservation"]: Reservation(
            row["reservation"],
            row["family"],
            row["ride"],
            int(row["seats"]),
            row["state"],
            moment(row["issued_at"]),
        )
        for row in _read("reserves", directory)
    }
    activations = {
        row["trigger"]: Activation(
            row["trigger"],
            row["group"],
            row["scope"],
            clock(row["deadline"]),
            tuple(row["applies_to"].split("|")),
            row["facts"],
        )
        for row in _read("activation", directory)
    }
    notices = {
        row["notice"]: Notice(
            row["notice"],
            row["ride"],
            row["state"],
            moment(row["effective_at"]),
            moment(row["received_at"]),
        )
        for row in _read("notices", directory)
    }

    by_trip: dict[str, dict[str, Observation]] = defaultdict(dict)
    for row in _read("observations", directory):
        elapsed = row["elapsed_minutes"]
        obs = Observation(
            row["record"],
            row["trip"],
            row["day"],
            row["corridor"],
            row["segment"],
            row["weather"],
            row["demand"],
            int(row["scheduled_minutes"]),
            None if elapsed == "" else int(elapsed),
            row["outcome"],
            row["mode"],
        )
        if obs.segment in by_trip[obs.trip]:
            raise SourceError(f"{obs.trip} 的 {obs.segment} 段重复记录")
        by_trip[obs.trip][obs.segment] = obs

    trips: dict[str, TripObservation] = {}
    for trip, segments in by_trip.items():
        first = next(iter(segments.values()))
        days = {obs.day for obs in segments.values()}
        backgrounds = {(obs.weather, obs.demand) for obs in segments.values()}
        if len(days) > 1:
            raise SourceError(f"{trip} 跨日: {sorted(days)}")
        if len(backgrounds) > 1:
            raise SourceError(f"{trip} 内部天气/客流背景不一致: {backgrounds}")
        weather, demand = backgrounds.pop()
        trips[trip] = TripObservation(trip, first.day, weather, demand, segments)

    return SourceData(slots, rides, walks, families, reservations, activations, notices, trips)


def notice_state(data: SourceData, ride: str, at: datetime) -> str:
    """Notice state in force at ``at``.

    The contract says the operating check follows the notice's effective time,
    not the order notices were received, so receipt time only breaks ties
    between notices that take effect at the same moment.
    """
    relevant = [
        notice
        for notice in data.notices.values()
        if notice.ride == ride and notice.effective_at <= at
    ]
    if not relevant:
        return "running"
    latest = max(relevant, key=lambda n: (n.effective_at, n.received_at, n.notice))
    return latest.state


def notice_timeline(data: SourceData) -> tuple[Notice, ...]:
    """Notices in force order, effective time first."""
    return tuple(sorted(data.notices.values(), key=lambda n: (n.effective_at, n.received_at, n.notice)))


def running_rides(data: SourceData, at: datetime) -> tuple[Ride, ...]:
    return tuple(r for r in data.rides.values() if notice_state(data, r.ride, at) == "running")
