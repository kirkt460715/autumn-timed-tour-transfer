"""Seat ledger: whole-party seats, kept promises, and emergency holds.

Every family travels on one ticket, so a ride is given the whole party or
nothing.  ``capacity`` counts people, and a family occupies ``people`` places.

A promise already made to a family (``state=locked``) is never taken away to
make room for someone else.  If the family is re-assigned away from that ride
the promise stays on the ride until the officer handles it, so it keeps using
capacity.  If the family is assigned to the very ride it was promised, the
promise is part of that family's party size and is not counted twice.

Two rules from the contract also live here:

* Adjacent segments on one corridor fail together, so one held seat cannot be
  the guarantee for two concurrent impacts on that corridor.
* Mutually exclusive contingencies may share a held seat only when their
  trigger and their go/no-go deadline distinguish them.  Two triggers in one
  exclusive group with the same deadline cannot be told apart in time, so they
  cannot share a seat.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from .model import Activation, Family, SourceData, hhmm, notice_state
from .risk import Plan


@dataclass(frozen=True)
class SeatClaim:
    family: str
    ride: str
    seats: int
    kind: str  # "locked" | "proposed"
    reservation: str | None = None


@dataclass(frozen=True)
class RideLoad:
    ride: str
    demand: int
    capacity: int
    families: tuple[str, ...]
    promised: tuple[str, ...]

    @property
    def free(self) -> int:
        return self.capacity - self.demand

    @property
    def shortfall(self) -> int:
        return max(0, self.demand - self.capacity)

    def describe(self) -> str:
        who = "、".join(self.families) or "无家庭"
        return f"{self.ride} 需 {self.demand} 席、可容 {self.capacity} 席，差 {self.shortfall} 席；涉及 {who}"


def running(data: SourceData, at: datetime) -> tuple[str, ...]:
    return tuple(ride for ride in data.rides if notice_state(data, ride, at) == "running")


def locked_claims(
    data: SourceData,
    *,
    at: datetime | None = None,
    include_stopped: bool = False,
    released: Iterable[str] = (),
) -> tuple[SeatClaim, ...]:
    """Promises made to families, optionally restricted to rides still running.

    ``released`` names reservations the officer has already released with the
    family's agreement.  They stop using capacity from that point on; nothing
    here releases a promise on its own.
    """
    released = set(released)
    out = []
    for reserve in data.reservations.values():
        if reserve.state != "locked" or reserve.reservation in released:
            continue
        if not include_stopped and at is not None and notice_state(data, reserve.ride, at) != "running":
            continue
        out.append(SeatClaim(reserve.family, reserve.ride, reserve.seats, "locked", reserve.reservation))
    return tuple(out)


def promised_seats(data: SourceData, family: str, ride: str) -> int:
    return sum(
        claim.seats
        for claim in locked_claims(data, include_stopped=True)
        if claim.family == family and claim.ride == ride
    )


def claims_for(data: SourceData, family: Family, plan: Plan) -> tuple[SeatClaim, ...]:
    """Seats this plan asks for, net of what the family is already promised."""
    out = []
    for ride in plan.bus_rides():
        already = promised_seats(data, family.family, ride)
        extra = max(0, family.people - already)
        if extra:
            out.append(SeatClaim(family.family, ride, extra, "proposed"))
    return tuple(out)


def loads(
    data: SourceData,
    assignment: Mapping[str, Plan],
    *,
    at: datetime | None = None,
    released: Iterable[str] = (),
) -> dict[str, RideLoad]:
    """Demand on every ride, counting kept promises exactly once."""
    assigned_on: dict[str, set[str]] = defaultdict(set)
    for family_id, plan in assignment.items():
        for ride in plan.bus_rides():
            assigned_on[ride].add(family_id)

    out: dict[str, RideLoad] = {}
    for ride, spec in sorted(data.rides.items()):
        families = sorted(assigned_on.get(ride, set()))
        demand = sum(data.families[f].people for f in families)
        promised: list[str] = []
        for claim in locked_claims(data, at=at, released=released):
            if claim.ride != ride or claim.family in families:
                continue
            demand += claim.seats
            promised.append(claim.family)
        out[ride] = RideLoad(ride, demand, spec.capacity, tuple(families), tuple(sorted(promised)))
    return out


def conflicts(
    data: SourceData,
    assignment: Mapping[str, Plan],
    *,
    at: datetime | None = None,
    released: Iterable[str] = (),
) -> tuple[RideLoad, ...]:
    return tuple(
        load for load in loads(data, assignment, at=at, released=released).values() if load.shortfall
    )


def accessible_ok(data: SourceData, family: Family, plan: Plan) -> bool:
    """Only a step-free family is barred from a ride marked inaccessible."""
    if family.access != "step_free":
        return True
    return all(data.rides[ride].accessible for ride in plan.bus_rides())


def feasible(
    data: SourceData,
    assignment: Mapping[str, Plan],
    *,
    at: datetime | None = None,
    released: Iterable[str] = (),
) -> tuple[bool, tuple[RideLoad, ...]]:
    allowed = set(running(data, at)) if at is not None else set(data.rides)
    for family_id, plan in assignment.items():
        if not accessible_ok(data, data.families[family_id], plan):
            return False, ()
        if not set(plan.bus_rides()) <= allowed:
            return False, ()
    found = conflicts(data, assignment, at=at, released=released)
    return not found, found


def free_seats(
    data: SourceData,
    ride: str,
    assignment: Mapping[str, Plan],
    *,
    at: datetime | None = None,
) -> int:
    return loads(data, assignment, at=at)[ride].free


def recovery_pressure(
    data: SourceData,
    assignment: Mapping[str, Plan],
    alternatives: Mapping[str, Iterable[Plan]],
    *,
    at: datetime | None = None,
) -> dict[str, tuple[str, ...]]:
    """Families whose remaining recovery plans are squeezed out by this load.

    For each family, look at the other plans it could still use and report the
    rides those plans need but can no longer take the whole party.  The family's
    own current seats are released before the check, because moving is exactly
    what recovery means; seats promised to *other* families stay counted.
    """
    out: dict[str, tuple[str, ...]] = {}
    for family_id, family in data.families.items():
        without = {f: p for f, p in assignment.items() if f != family_id}
        current_loads = loads(data, without, at=at)
        current = assignment.get(family_id)
        blocked: set[str] = set()
        for plan in alternatives.get(family_id, ()):
            if plan == current:
                continue
            for ride in plan.bus_rides():
                needed = max(0, family.people - promised_seats(data, family_id, ride))
                if current_loads[ride].free < needed:
                    blocked.add(ride)
        if blocked:
            out[family_id] = tuple(sorted(blocked))
    return out


def exclusive_pairs(data: SourceData) -> tuple[tuple[Activation, Activation], ...]:
    exclusives = sorted(
        (a for a in data.activations.values() if a.scope == "exclusive"),
        key=lambda a: a.trigger,
    )
    out = []
    for i, first in enumerate(exclusives):
        for second in exclusives[i + 1 :]:
            if first.group == second.group and first.trigger != second.trigger:
                out.append((first, second))
    return tuple(out)


def shared_seat_problems(data: SourceData) -> tuple[str, ...]:
    """Exclusive triggers that want one seat and cannot be told apart in time."""
    out = []
    for first, second in exclusive_pairs(data):
        if first.deadline == second.deadline:
            out.append(
                f"{first.trigger} 与 {second.trigger} 同属互斥组 {first.group}，"
                f"去留截止同为 {hhmm(first.deadline)}；届时才知道是哪一种，"
                "同一应急席无法区分，不能共用"
            )
        else:
            out.append(
                f"{first.trigger}（截止 {hhmm(first.deadline)}）与 "
                f"{second.trigger}（截止 {hhmm(second.deadline)}）同属互斥组 {first.group}；"
                "可按较早的截止区分，但共用席位须按较早截止留席"
            )
    return tuple(out)


def concurrent_corridor_impacts(data: SourceData) -> tuple[str, ...]:
    """Shared triggers that put more than one family on one corridor at once.

    Adjacent segments on one corridor fail together, so when one triggered
    impact already covers several families, each of them needs its own seats on
    that corridor.  A single held seat block cannot serve them all, which is
    exactly what has to be said out loud rather than counted once.
    """
    by_corridor: dict[str, list[Activation]] = defaultdict(list)
    for trigger in data.activations.values():
        if trigger.scope != "shared":
            continue
        corridors = [c for c in ("AB", "BC") if c in trigger.group.split("_")]
        if len(corridors) != 1:
            raise ValueError(f"{trigger.trigger} 的共享组 {trigger.group} 未标明唯一走廊")
        by_corridor[corridors[0]].append(trigger)
    out = []
    for corridor, triggers in sorted(by_corridor.items()):
        triggers = sorted(triggers, key=lambda t: (t.deadline, t.trigger))
        families = sorted({f for t in triggers for f in t.applies_to})
        if len(triggers) < 2 and len(families) < 2:
            continue
        triggers_text = "、".join(t.trigger for t in triggers)
        out.append(
            f"{corridor} 走廊的共享触发 {triggers_text} 同时覆盖 {'、'.join(families)}："
            "相邻段一起变慢时，一车席位不能同时作为多个并发影响的保证，须逐户留席"
        )
    return tuple(out)


def triggers_for(data: SourceData, family: str) -> tuple[Activation, ...]:
    return tuple(
        sorted(
            (a for a in data.activations.values() if family in a.applies_to),
            key=lambda a: (a.deadline, a.trigger),
        )
    )


def admission_holds(data: SourceData) -> tuple[tuple[Activation, ...], ...]:
    """Exclusive triggers that would need one seat at the same deadline."""
    grouped: dict[tuple[str, int], list[Activation]] = defaultdict(list)
    for trigger in data.activations.values():
        if trigger.scope != "exclusive":
            continue
        grouped[(trigger.group, trigger.deadline)].append(trigger)
    return tuple(
        tuple(sorted(v, key=lambda t: t.trigger))
        for v in grouped.values()
        if len(v) > 1
    )


def unused_promises(
    data: SourceData,
    assignment: Mapping[str, Plan],
    *,
    at: datetime | None = None,
) -> tuple[SeatClaim, ...]:
    """Promises still holding seats on a ride the family's new plan does not use.

    These are not conflicts and are not released automatically: the family was
    already told it held them.  They are listed so the officer can decide
    whether to release the seat, and so the briefing can say how much capacity
    that decision would free.
    """
    out = []
    for claim in locked_claims(data, at=at, include_stopped=True):
        plan = assignment.get(claim.family)
        if plan is None:
            out.append(claim)
        elif claim.ride not in plan.bus_rides():
            out.append(claim)
    return tuple(out)
