"""Revision workflow for the duty officer.

A run sheet is prepared before the officer signs it off.  While it is being
prepared an operating notice can arrive.  A sheet worked out against the old
timetable must not be used to claim seats once a notice has taken effect: the
plan is stale and has to be rebuilt against the rides that actually run.

Seats already promised to a family are different.  When a notice hits a ride a
family already holds, the family keeps what it was told.  Each family is put in
one of three buckets for the officer:

``keep``         the promised arrangement still works unchanged;
``confirm``      the promise can be met, but on different terms the family must
                 be told about (another ride, another slot, a walk leg);
``unrescuable``  the promise cannot be met and no replacement clears the
                 family floor - the officer has to decide.

The officer also sees, separately, the families that could change venue on
their own but have no shuttle guarantee at all.  Those are not seat conflicts;
they are unsecured self-connections and are reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from . import risk, seats
from .model import SourceData, notice_state
from .risk import Assessment, Plan

DRAFT = "draft"
CONFIRMED = "confirmed"
STALE = "stale"

KEEP = "keep"
CONFIRM = "confirm"
UNRESCUABLE = "unrescuable"


@dataclass(frozen=True)
class Draft:
    """A prepared run sheet, not yet signed off."""

    assignment: Mapping[str, Plan]
    prepared_at: datetime
    state: str = DRAFT
    confirmed_at: datetime | None = None


@dataclass(frozen=True)
class NoticeImpact:
    notice: str
    ride: str
    effective_at: datetime
    received_at: datetime
    affected: tuple[str, ...]

    def describe(self) -> str:
        who = "、".join(self.affected) or "无已排家庭"
        return (
            f"{self.notice} {self.ride} 自 {self.effective_at:%H:%M} 起停开"
            f"（收件 {self.received_at:%H:%M}），涉及 {who}"
        )


@dataclass(frozen=True)
class FamilyDisposition:
    family: str
    bucket: str
    current: Plan
    replacement: Plan | None
    reasons: tuple[str, ...]
    kept_reservations: tuple[str, ...] = ()

    def describe(self) -> str:
        target = self.replacement.label() if self.replacement else "无达标替代"
        tail = f"；{'；'.join(self.reasons)}" if self.reasons else ""
        return f"{self.family}：现 {self.current.label()} → {target}{tail}"


@dataclass(frozen=True)
class SelfConnect:
    """A family walking a corridor with no shuttle guarantee on it."""

    family: str
    corridor: str
    minutes: int
    bus_option: Plan | None
    bus_option_free: bool

    def describe(self) -> str:
        if self.bus_option is None:
            tail = "这条走廊没有可留席的班次"
        elif self.bus_option_free:
            tail = f"可改留席 {self.bus_option.label()}"
        else:
            tail = f"备选 {self.bus_option.label()} 已无整组空席"
        return (
            f"{self.family} 在 {self.corridor} 走廊自行步行 {self.minutes} 分钟换乘"
            f"（无接驳保证），{tail}"
        )


def check_staleness(data: SourceData, draft: Draft, at: datetime) -> str:
    """Whether a draft is still usable at ``at``.

    A notice effective after the draft was prepared and on or before ``at``
    makes the seat claims stale even if the officer has not confirmed yet.
    """
    if draft.state == CONFIRMED:
        return CONFIRMED
    for notice in data.notices.values():
        if draft.prepared_at < notice.effective_at <= at:
            if any(notice.ride in plan.bus_rides() for plan in draft.assignment.values()):
                return STALE
    return DRAFT


def notice_impacts(
    data: SourceData, at: datetime, draft: Draft | None = None
) -> tuple[NoticeImpact, ...]:
    """Notices in force by ``at`` and the families whose seats they touch."""
    out = []
    ordered = sorted(data.notices.values(), key=lambda n: (n.effective_at, n.received_at, n.notice))
    for notice in ordered:
        if notice.effective_at > at:
            continue
        affected = {
            family
            for family, plan in (draft.assignment.items() if draft else ())
            if notice.ride in plan.bus_rides()
        }
        out.append(
            NoticeImpact(
                notice.notice, notice.ride, notice.effective_at, notice.received_at, tuple(sorted(affected))
            )
        )
    return tuple(out)


def rebuild(data: SourceData, at: datetime, families: Iterable[str] | None = None) -> dict[str, Assessment]:
    """Best floor-meeting plan per family against the rides running at ``at``."""
    usable = seats.running(data, at)
    out: dict[str, Assessment] = {}
    for family_id in families or data.families:
        family = data.families[family_id]
        best: Assessment | None = None
        for plan in risk.candidate_plans(data, family, usable_rides=usable):
            assessed = risk.evaluate(data, family, plan)
            if best is None or _stronger(assessed, best):
                best = assessed
        if best is not None:
            out[family_id] = best
    return out


def _stronger(candidate: Assessment, incumbent: Assessment) -> bool:
    return (
        candidate.meets_floor,
        candidate.estimate,
        candidate.optimistic,
        -len(candidate.phantom_rides()),
        candidate.evidence,
    ) > (
        incumbent.meets_floor,
        incumbent.estimate,
        incumbent.optimistic,
        -len(incumbent.phantom_rides()),
        incumbent.evidence,
    )


def dispositions(data: SourceData, draft: Draft, at: datetime) -> tuple[FamilyDisposition, ...]:
    """Sort each family in a draft into keep / confirm / unrescuable at ``at``."""
    rebuilt = rebuild(data, at, draft.assignment.keys())
    out: list[FamilyDisposition] = []
    for family_id, current in sorted(draft.assignment.items()):
        family = data.families[family_id]
        promised = tuple(
            claim.reservation
            for claim in seats.locked_claims(data, at=at, include_stopped=True)
            if claim.family == family_id
        )
        stopped = [ride for ride in current.bus_rides() if notice_state(data, ride, at) != "running"]
        alternative = rebuilt.get(family_id)
        if not stopped:
            out.append(FamilyDisposition(family_id, KEEP, current, current, (), promised))
            continue
        reasons = [f"{ride} 自通知生效后停开" for ride in stopped]
        if alternative is None or not alternative.meets_floor:
            floor = alternative.estimate if alternative else 0.0
            reasons.append(f"至 {at:%H:%M} 无达底线 {family.floor:.2f} 的替代（最好 {floor:.2f}）")
            out.append(FamilyDisposition(family_id, UNRESCUABLE, current, None, tuple(reasons), promised))
            continue
        replacement = alternative.plan
        if replacement == current:
            out.append(FamilyDisposition(family_id, KEEP, current, current, (), promised))
            continue
        reasons.append(
            f"替代 {replacement.label()} 完成率 {alternative.estimate:.2f}"
            f"（区间 {alternative.scenario_range[0]:.2f}-{alternative.scenario_range[1]:.2f}）"
        )
        out.append(FamilyDisposition(family_id, CONFIRM, current, replacement, tuple(reasons), promised))
    return tuple(out)


def self_connects(data: SourceData, assignment: Mapping[str, Plan], at: datetime) -> tuple[SelfConnect, ...]:
    """Families walking a corridor on their own with no shuttle guarantee there.

    A walk leg in the plan is a self-connection the family makes without a seat.
    It is acceptable only where the family can actually walk it, and it must be
    surfaced, because the walk is now part of what the family was told.  Where a
    seat could still be held, the empty-load check says whether that is even
    possible under the promises already made.
    """
    empty_loads = seats.loads(data, {}, at=at)
    riding = seats.running(data, at)
    out = []
    for family_id, plan in sorted(assignment.items()):
        family = data.families[family_id]
        promised = {
            claim.ride for claim in seats.locked_claims(data, at=at, include_stopped=True) if claim.family == family_id
        }
        for corridor, mode in (("AB", plan.ab), ("BC", plan.bc)):
            if mode != risk.WALK:
                continue
            minutes = risk.walk_minutes(data, family, corridor)
            if minutes is None:
                continue
            if any(ride.corridor == corridor and ride.ride in promised for ride in data.corridor_rides(corridor)):
                continue
            bus_option = None
            for candidate in risk.candidate_plans(data, family, usable_rides=riding):
                if candidate.slots != plan.slots:
                    continue
                if corridor == "AB" and candidate.ab == risk.WALK:
                    continue
                if corridor == "BC" and candidate.bc == risk.WALK:
                    continue
                bus_option = candidate
                break
            affordable = bool(
                bus_option
                and all(
                    empty_loads[ride].free >= max(0, family.people - seats.promised_seats(data, family_id, ride))
                    for ride in bus_option.bus_rides()
                )
            )
            out.append(SelfConnect(family_id, corridor, minutes, bus_option, affordable))
    return tuple(out)


def stranded_promises(data: SourceData, at: datetime) -> dict[str, tuple[str, ...]]:
    """Promises sitting on rides a notice has stopped by ``at``."""
    out: dict[str, list[str]] = {}
    for reserve in data.reservations.values():
        if reserve.state != "locked":
            continue
        if notice_state(data, reserve.ride, at) != "running":
            out.setdefault(reserve.family, []).append(reserve.ride)
    return {family: tuple(rides) for family, rides in sorted(out.items())}


def kept_promises(data: SourceData, at: datetime) -> dict[str, tuple[str, ...]]:
    """Promises on rides still running at ``at``."""
    out: dict[str, list[str]] = {}
    for claim in seats.locked_claims(data, at=at):
        out.setdefault(claim.family, []).append(claim.ride)
    return {family: tuple(rides) for family, rides in sorted(out.items())}
