"""Compare the two levers a line operator has: change the slot, or hold a seat.

Both levers are judged on the same evidence as everything else: whole-journey
completion on the historical trips that actually observed the legs involved.
Every option is re-assessed from scratch, so no figure is carried over from a
plan with a different evidence set.

The comparison also looks sideways.  Holding a seat on a shared shuttle can fill
a ride another family was counting on for recovery, and slots have a fixed
number of spare places.  Each option therefore reports

* the families that share its rides or its moved slots;
* whether the option leaves those families any later recovery;
* the observation that makes it fail, not just how often it failed.

That last point is the "which segment, which full ride, which closing window"
answer the officer needs; a bare percentage cannot be acted on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from . import risk, seats
from .model import Family, SourceData, notice_state
from .risk import Assessment, Plan


@dataclass(frozen=True)
class Option:
    """One way of absorbing the problem for a family."""

    kind: str  # "slot" | "seat"
    plan: Plan
    assessment: Assessment
    cohort: tuple[str, ...]
    blocked_recovery: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.assessment.meets_floor

    @property
    def floor(self) -> float:
        return self.assessment.floor

    def describe(self) -> str:
        head = "改场次" if self.kind == "slot" else "留接驳席"
        low, high = self.assessment.scenario_range
        return (
            f"[{head}] {self.plan.label()} 完成率 {self.assessment.estimate:.2f}"
            f"（支持区间 {low:.2f}-{high:.2f}，n={self.assessment.evidence}）"
            f"{'，达标' if self.feasible else '，未达底线'}"
        )


def family_rides(data: SourceData, family: Family) -> tuple[str, ...]:
    """Rides this family is attached to: booked legs plus held promises."""
    booked = set(family.trip)
    out: list[str] = []
    for corridor in ("AB", "BC"):
        for ride in data.corridor_rides(corridor):
            if ride.ride in booked:
                out.append(ride.ride)
    for claim in seats.locked_claims(data, include_stopped=True):
        if claim.family == family.family and claim.ride not in out:
            out.append(claim.ride)
    return tuple(out)


def _cohort(data: SourceData, family: Family, plan: Plan) -> tuple[str, ...]:
    """Other families sharing this plan's rides or being pushed off its slots."""
    out = set()
    for other in data.families.values():
        if other.family == family.family:
            continue
        if set(family_rides(data, other)) & set(plan.bus_rides()):
            out.add(other.family)
        for index, slot in enumerate(plan.slots):
            if slot != family.trip[index] and slot in other.moves[index]:
                out.add(other.family)
    return tuple(sorted(out))


def current_plan(data: SourceData, family: Family) -> Plan:
    """Public name for the family's booked arrangement."""
    return _booked_plan(data, family)


def _corridor_mode(data: SourceData, family: Family, corridor: str) -> str:
    """A promise on this corridor if there is one, else the permitted walk or first ride.

    A missing promise means no seat was ever promised on that corridor; a walk
    is shown only where the family may actually take it, and otherwise the
    corridor's first ride stands in as the baseline the family would use.
    """
    for claim in seats.locked_claims(data, include_stopped=True):
        if claim.family == family.family and data.rides[claim.ride].corridor == corridor:
            return claim.ride
    if risk.walk_capable(data, family, corridor):
        return risk.WALK
    return _first_ride(data, corridor)


def _booked_plan(data: SourceData, family: Family) -> Plan:
    """The family's current arrangement: booked slots, and the ride held per corridor."""
    return Plan(
        family.family,
        family.trip,
        _corridor_mode(data, family, "AB"),
        _corridor_mode(data, family, "BC"),
    )


def _first_ride(data: SourceData, corridor: str) -> str:
    rides = data.corridor_rides(corridor)
    if not rides:  # pragma: no cover - the source always has one
        raise ValueError(f"{corridor} 走廊没有班次")
    return rides[0].ride


def compare(
    data: SourceData,
    family: Family,
    *,
    at: datetime,
    baseline: Mapping[str, Plan] | None = None,
    stratum: str = risk.BASELINE,
    usable_rides: Iterable[str] | None = None,
) -> tuple[Option, ...]:
    """Assess slot changes and seat holds for one family on the same evidence.

    ``baseline`` is the rest of the day's arrangement.  Each option is scored
    with the rest of the day held fixed, so "who loses a recovery chance" is
    about the other families, not about this one's old seats.
    """
    usable = tuple(usable_rides) if usable_rides is not None else seats.running(data, at)
    current = _booked_plan(data, family)
    rest = {fid: plan for fid, plan in (baseline or {}).items() if fid != family.family}
    if not rest:
        rest = _rest_defaults(data, family, at)
    options: list[Option] = []
    seen: set[tuple[str, ...]] = set()

    for plan in risk.candidate_plans(data, family, usable_rides=usable):
        if plan.key() in seen:
            continue
        seen.add(plan.key())
        assessed = risk.evaluate(data, family, plan, stratum=stratum)
        moved = plan.slots != family.trip
        kind = "slot" if moved else "seat"
        assignment = dict(rest)
        assignment[family.family] = plan
        pressure = seats.recovery_pressure(
            data,
            assignment,
            {fid: (p,) for fid, p in rest.items()},
            at=at,
        )
        blocked = {
            fid: tuple(rides)
            for fid, rides in pressure.items()
            if fid != family.family and rides
        }
        notes = list(assessed.notes)
        for load in seats.conflicts(data, assignment, at=at):
            notes.append(load.describe())
        options.append(
            Option(
                kind=kind,
                plan=plan,
                assessment=assessed,
                cohort=_cohort(data, family, plan),
                blocked_recovery=tuple(f"{fid}:{'、'.join(rides)}" for fid, rides in sorted(blocked.items())),
                notes=tuple(notes),
            )
        )
    return tuple(options)


def _rest_defaults(data: SourceData, family: Family, at: datetime) -> dict[str, Plan]:
    """Other families' current arrangements, so a lone comparison still looks sideways."""
    return {
        other.family: _booked_plan(data, other)
        for other in data.families.values()
        if other.family != family.family
    }


def shared_impact(
    data: SourceData,
    assignment: Mapping[str, Plan],
    *,
    at: datetime,
) -> dict[str, tuple[str, ...]]:
    """Rides where one hold would be asked to cover concurrent affected families."""
    loads = seats.loads(data, assignment, at=at)
    by_group: dict[str, set[str]] = {}
    for trigger in data.activations.values():
        if trigger.scope == "shared":
            by_group.setdefault(trigger.group, set()).update(trigger.applies_to)
    out: dict[str, tuple[str, ...]] = {}
    for ride, load in loads.items():
        for group, families in by_group.items():
            both = tuple(sorted(set(families) & set(load.families)))
            if len(both) > 1:
                out[ride] = both
    return out


def failure_reasons(
    data: SourceData,
    family: Family,
    plan: Plan,
    *,
    at: datetime,
) -> tuple[str, ...]:
    """Reasons a plan stops working at ``at``, named by segment and ride.

    "The car was full" and "the transfer window had already closed" lead to
    different actions, so they are listed separately.
    """
    reasons: list[str] = []
    for ride in plan.bus_rides():
        if notice_state(data, ride, at) != "running":
            reasons.append(f"{ride} 因通知停开，不能按旧时刻占座")
    assessment = risk.evaluate(data, family, plan)
    for result in assessment.outcomes:
        if result.outcome == "completed":
            continue
        if result.cause == "late_B":
            mark = result.marks[-1]
            reasons.append(f"{result.trip} 观测：到 B 馆 {mark // 60:02d}:{mark % 60:02d}，换场窗口已关（最迟签到已过）")
            break
        if result.cause == "late_C":
            mark = result.marks[-1]
            reasons.append(f"{result.trip} 观测：到 C 馆 {mark // 60:02d}:{mark % 60:02d}，换场窗口已关（最迟签到已过）")
            break
        if result.cause.startswith("miss_"):
            leg = result.cause.split("_", 1)[1].upper()
            ride = plan.ab if leg == "AB" else plan.bc
            window = _window_text(data, plan, leg, result)
            reasons.append(
                f"{result.trip} 观测：{leg} 走廊 {ride} 到站 {window} 已发车，车上无位"
            )
            break
        if result.cause.endswith("_cancelled"):
            leg = result.cause.split("_")[0].upper()
            reasons.append(f"{result.trip} 观测：{leg} 走廊班次停开，旅程没有走完")
            break
    return tuple(reasons)


def _window_text(data: SourceData, plan: Plan, leg: str, result) -> str:
    """The departure the family actually missed, as a clock time."""
    ride = plan.ab if leg == "AB" else plan.bc
    if ride == risk.WALK or ride not in data.rides:
        return "（步行段无发车时刻）"
    leaves = data.rides[ride].leaves
    return f"{leaves // 60:02d}:{leaves % 60:02d}"
