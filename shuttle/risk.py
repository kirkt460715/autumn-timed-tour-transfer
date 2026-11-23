"""Whole-journey completion risk for one family on one candidate plan.

The unit of evidence is a historical trip, not a segment.  On one observed trip
the two venue visits and the connecting legs were recorded together, so they
share that day's weather and demand.  Treating the legs as independent and
multiplying per-segment success rates would manufacture confidence that the
source does not support; instead every observed trip is simulated end to end
and counted once.

Which trips can speak to a plan
    A trip is evidence for a plan only if every leg the plan actually rides was
    recorded that day.  A walked corridor needs no record.  Trips missing the
    record are left out instead of being filled in, so a plan whose buses were
    never observed together has a smaller, explicitly stated evidence set.

Unobserved walk legs
    A walk has registered minutes but no elapsed observation, so it is reported
    as a range: the optimistic bound uses the registered minutes, the
    pessimistic bound adds the worst positive connecting-leg delay observed on
    the same corridor that day.  Adjacent segments on one corridor slow down
    together, so a day where the bus leg ran late is exactly the day the walk
    would too.  When the connecting leg was cancelled or unrecorded there is no
    delay to carry over, and the day contributes the registered minutes only.
    The lower bound is what must clear the family floor.

Evidence buckets
    ``completed``      reached every last-check-in minute;
    ``known_late``     reached a check-in too late, or missed a departure;
    ``not_completed``  a connecting service was cancelled and no walk rescued it;
    ``incomplete``     the source observation never finished: censored, counted
                       neither as a pass nor as a known failure.
    A trip whose segments all ran and all ran within schedule is genuine
    zero-lateness evidence and counts as a completion, never as censored.

Session timing
    A visit is a fixed booking window.  It ends at the slot's ``ends`` plus the
    observed overrun; a late entry shortens the visit but does not extend the
    window.  The next departure is checked against that exit, and each check-in
    bound is inclusive (``arrival <= last_checkin``).

The floor in ``family_route`` is a punctuality agreement for the normally
operating line.  Rain is a separate contingency layer driven by the activation
triggers, so it is reported beside the floor, not folded into it.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .model import Family, SourceData, TripObservation, hhmm

WALK = "walk"
BASELINE = "normal"
CONTINGENCY = "rain"


@dataclass(frozen=True)
class Plan:
    """A slot combination plus one mode for each connecting corridor."""

    family: str
    slots: tuple[str, str, str]
    ab: str
    bc: str

    def modes(self) -> tuple[str, str]:
        return self.ab, self.bc

    def has_walk(self) -> bool:
        return WALK in self.modes()

    def bus_rides(self) -> tuple[str, ...]:
        return tuple(mode for mode in self.modes() if mode != WALK)

    def self_connect(self) -> bool:
        """The whole journey can be walked, so it needs no shuttle seat."""
        return not self.bus_rides()

    def label(self) -> str:
        ab = "步行" if self.ab == WALK else self.ab
        bc = "步行" if self.bc == WALK else self.bc
        return f"{'|'.join(self.slots)} · AB:{ab} · BC:{bc}"

    def with_modes(self, ab: str, bc: str) -> "Plan":
        return Plan(self.family, self.slots, ab, bc)

    def key(self) -> tuple[str, ...]:
        return (self.family, *self.slots, self.ab, self.bc)


@dataclass(frozen=True)
class TripResult:
    trip: str
    day: str
    weather: str
    demand: str
    outcome: str
    cause: str
    boarded: tuple[str, ...]
    fallback: bool
    marks: tuple[int, ...]

    def describe(self) -> str:
        marks = " → ".join(hhmm(mark) for mark in self.marks)
        return f"{self.trip}({self.day}) {self.outcome}/{self.cause} {marks}"


@dataclass(frozen=True)
class Assessment:
    plan: Plan
    stratum: str
    counts: Counter
    outcomes: tuple[TripResult, ...]
    estimate: float
    optimistic: float
    ci: tuple[float, float]
    support: str
    scenario_range: tuple[float, float]
    floor: float
    meets_floor: bool
    fragile: bool
    notes: tuple[str, ...]

    @property
    def completed(self) -> int:
        return self.counts["completed"]

    @property
    def evidence(self) -> int:
        return self.completed + self.counts["known_late"] + self.counts["not_completed"]

    @property
    def censored(self) -> int:
        return self.counts["incomplete"]

    @property
    def margin(self) -> float:
        return self.estimate - self.floor

    @property
    def fallback_completions(self) -> int:
        """Trips that were on time only because a walk replaced a missed ride."""
        return sum(1 for o in self.outcomes if o.outcome == "completed" and o.fallback)

    def boarded_rides(self) -> set[str]:
        return {ride for o in self.outcomes for ride in o.boarded}

    def phantom_rides(self) -> tuple[str, ...]:
        """Booked rides the family never actually boards in any observed trip."""
        return tuple(ride for ride in self.plan.bus_rides() if ride not in self.boarded_rides())

    def causes(self) -> dict[str, int]:
        out: Counter = Counter()
        for result in self.outcomes:
            if result.outcome != "completed":
                out[result.cause] += 1
        return dict(sorted(out.items(), key=lambda item: (-item[1], item[0])))

    def failures(self) -> dict[str, int]:
        return {
            "known_late": self.counts["known_late"],
            "not_completed": self.counts["not_completed"],
            "censored": self.counts["incomplete"],
        }

    def witness(self, *causes: str) -> TripResult | None:
        """First trip that actually failed for one of the given reasons."""
        for result in self.outcomes:
            if result.outcome == "completed":
                continue
            if any(result.cause == cause or result.cause.startswith(cause) for cause in causes):
                return result
        return None


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def walk_minutes(data: SourceData, family: Family, corridor: str) -> int | None:
    """Registered minutes for a walk this family is allowed to take."""
    return data.walk_minutes(corridor, family.access)


def walk_capable(data: SourceData, family: Family, corridor: str) -> bool:
    return walk_minutes(data, family, corridor) is not None


def modes_for(
    data: SourceData,
    family: Family,
    corridor: str,
    usable: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Modes open to this family on a corridor: a permitted walk, then rides."""
    allowed = set(usable) if usable is not None else None
    out: list[str] = []
    if walk_capable(data, family, corridor):
        out.append(WALK)
    for ride in data.corridor_rides(corridor):
        if allowed is not None and ride.ride not in allowed:
            continue
        if family.access == "step_free" and not ride.accessible:
            continue
        out.append(ride.ride)
    return tuple(out)


def candidate_plans(
    data: SourceData,
    family: Family,
    *,
    usable_rides: Iterable[str] | None = None,
) -> tuple[Plan, ...]:
    """Every plan the family may be put on, honouring move range and spare seats.

    Moving to another slot needs ``spare`` places in that slot for the whole
    party; the booked slot is already sold and needs no spare.  One ticket covers
    the whole family, so the party is never split.
    """
    ab_modes = modes_for(data, family, "AB", usable_rides)
    bc_modes = modes_for(data, family, "BC", usable_rides)
    booked = family.trip
    plans: list[Plan] = []
    for a in family.moves[0]:
        for b in family.moves[1]:
            for c in family.moves[2]:
                combo = (a, b, c)
                fits = True
                for index, slot in enumerate(combo):
                    if slot == booked[index]:
                        continue
                    if family.people > data.slots[slot].spare:
                        fits = False
                        break
                if not fits:
                    continue
                for ab in ab_modes:
                    for bc in bc_modes:
                        plans.append(Plan(family.family, combo, ab, bc))
    return tuple(plans)


def corridor_envelope(trip: TripObservation, corridor: str) -> int:
    """Worst positive delay on this corridor that day.

    Only the *connecting* legs are considered, because the venue visits' own
    overruns are already applied to the exit time; counting them again would
    double the delay.  Adjacent segments on one corridor slow down together, so
    the trip's own worst connecting delay is the stress a walk leg receives.  A
    cancelled or unrecorded connecting leg leaves nothing to carry: the walk is
    then stressed with 0 and the range just reflects the unobserved leg.
    """
    worst = 0
    for name in ("ab", "bc"):
        obs = trip.segments.get(name)
        if obs is None or obs.corridor != corridor or obs.elapsed_minutes is None:
            continue
        worst = max(worst, obs.deviation)
    return worst


def evidence_trips(data: SourceData, plan: Plan, stratum: str = BASELINE) -> tuple[TripObservation, ...]:
    """Trips that can speak to this plan: every ride leg it uses was recorded."""
    need = []
    if plan.ab != WALK:
        need.append("ab")
    if plan.bc != WALK:
        need.append("bc")
    return tuple(
        trip
        for trip in data.trips.values()
        if trip.weather == stratum and all(trip.state(segment) != "missing" for segment in need)
    )


def _connector(
    data: SourceData,
    family: Family,
    trip: TripObservation,
    *,
    corridor: str,
    mode: str,
    origin: int,
    stress: bool,
) -> tuple[str, int | None, str, bool, str | None]:
    """Arrival at the next venue.

    Returns ``(outcome, arrival, cause, used_fallback, boarded_ride)`` with
    ``outcome`` one of ``"ok"`` / ``"missed"`` / ``"unfinished"`` / ``"censored"``.

    ``missed`` means the family was there but the ride had gone or the visit ran
    over: a known lateness.  ``unfinished`` means the booked service was
    cancelled and no walk could take its place: the journey could not be
    completed at all, which is a different fact and is reported separately.
    A walk fallback is recorded, never silently absorbed.
    """
    prefix = "ab" if corridor == "AB" else "bc"
    cause = "ok"
    unfinished = False
    boarded: str | None = None

    if mode != WALK:
        ride = data.rides[mode]
        state = trip.state(prefix)
        if state == "observed":
            if origin <= ride.leaves:
                return "ok", ride.arrives + trip.segment(prefix).deviation, "ok", False, ride.ride  # type: ignore[union-attr]
            cause = f"miss_{prefix}"
        elif state == "cancelled":
            cause = f"{prefix}_cancelled"
            unfinished = True
        else:
            return "censored", None, f"{prefix}_incomplete", False, None

    if not walk_capable(data, family, corridor):
        return ("unfinished" if unfinished else "missed"), None, cause, False, None

    minutes = walk_minutes(data, family, corridor)
    assert minutes is not None
    if stress:
        minutes += corridor_envelope(trip, corridor)
    return "ok", origin + minutes, cause, mode != WALK, boarded


def simulate(
    data: SourceData,
    family: Family,
    plan: Plan,
    trip: TripObservation,
    *,
    stress: bool = False,
) -> TripResult:
    """Run one historical trip through one plan, end to end."""
    slots = [data.slots[s] for s in plan.slots]

    def result(
        outcome: str,
        cause: str,
        marks: list[int],
        boarded: tuple[str, ...] = (),
        fallback: bool = False,
    ) -> TripResult:
        return TripResult(trip.trip, trip.day, trip.weather, trip.demand, outcome, cause, boarded, fallback, tuple(marks))

    state_a = trip.state("a")
    if state_a in {"incomplete", "missing"}:
        return result("incomplete", f"a_{state_a}", [])
    if state_a == "cancelled":
        return result("not_completed", "a_cancelled", [])
    exit_a = max(slots[0].ends + trip.segment("a").deviation, slots[0].starts)  # type: ignore[union-attr]

    outcome, arrival_b, cause, fallback, boarded = _connector(
        data, family, trip, corridor="AB", mode=plan.ab, origin=exit_a, stress=stress
    )
    boarded_rides = (boarded,) if boarded else ()
    if outcome == "censored":
        return result("incomplete", cause, [exit_a], boarded_rides, fallback)
    if outcome in {"missed", "unfinished"}:
        bucket = "known_late" if outcome == "missed" else "not_completed"
        return result(bucket, cause, [exit_a], boarded_rides, fallback)
    assert arrival_b is not None
    if arrival_b > slots[1].last_checkin:
        return result("known_late", "late_B", [exit_a, arrival_b], boarded_rides, fallback)

    state_b = trip.state("b")
    if state_b in {"incomplete", "missing"}:
        return result("incomplete", f"b_{state_b}", [exit_a, arrival_b], boarded_rides, fallback)
    if state_b == "cancelled":
        return result("not_completed", "b_cancelled", [exit_a, arrival_b], boarded_rides, fallback)
    exit_b = max(
        slots[1].ends + trip.segment("b").deviation,  # type: ignore[union-attr]
        arrival_b,
        slots[1].starts,
    )

    outcome2, arrival_c, cause2, fallback2, boarded2 = _connector(
        data, family, trip, corridor="BC", mode=plan.bc, origin=exit_b, stress=stress
    )
    marks = [exit_a, arrival_b, exit_b]
    boarded_rides += (boarded2,) if boarded2 else ()
    used_fallback = fallback or fallback2
    if outcome2 == "censored":
        return result("incomplete", cause2, marks, boarded_rides, used_fallback)
    if outcome2 in {"missed", "unfinished"}:
        bucket = "known_late" if outcome2 == "missed" else "not_completed"
        return result(bucket, cause2, marks, boarded_rides, used_fallback)
    assert arrival_c is not None
    marks.append(arrival_c)
    if arrival_c > slots[2].last_checkin:
        return result("known_late", "late_C", marks, boarded_rides, used_fallback)
    return result("completed", "ok", marks, boarded_rides, used_fallback)


def _run(
    data: SourceData,
    family: Family,
    plan: Plan,
    stratum: str,
    *,
    stress: bool,
) -> tuple[Counter, list[TripResult]]:
    counts: Counter = Counter()
    results: list[TripResult] = []
    for trip in evidence_trips(data, plan, stratum):
        outcome = simulate(data, family, plan, trip, stress=stress)
        counts[outcome.outcome] += 1
        results.append(outcome)
    return counts, results


def _rate(counts: Counter) -> tuple[float, int]:
    evidence = counts["completed"] + counts["known_late"] + counts["not_completed"]
    return (counts["completed"] / evidence if evidence else 0.0), evidence


def evaluate(
    data: SourceData,
    family: Family,
    plan: Plan,
    *,
    stratum: str = BASELINE,
) -> Assessment:
    """Assess one plan, with an explicit supported range when a leg is unobserved.

    The reported ``estimate`` is the pessimistic bound; ``optimistic`` is the
    registered-minute bound.  ``meets_floor`` is decided on the pessimistic
    bound, so a route with unsupported walk legs cannot pass on optimism alone.
    """
    plain, plain_results = _run(data, family, plan, stratum, stress=False)
    optimistic, evidence = _rate(plain)

    if plan.has_walk():
        stressed, stressed_results = _run(data, family, plan, stratum, stress=True)
        pessimistic, _ = _rate(stressed)
        low, high = min(optimistic, pessimistic), max(optimistic, pessimistic)
        if pessimistic <= optimistic:
            counts, results = stressed, stressed_results
        else:  # no walk failure ever appears; the optimistic run already binds
            counts, results = plain, plain_results
        support = "insufficient" if evidence == 0 else "range"
        notes = ["含无用时观测的步行段：区间下限按同趟同走廊最大偏差加压"]
    else:
        low = high = optimistic
        counts, results = plain, plain_results
        support = "insufficient" if evidence == 0 else "joint"
        notes = []

    meetable = evidence > 0 and high >= family.floor
    meets = evidence > 0 and low >= family.floor
    fragile = bool(meetable and not meets)
    if evidence == 0:
        notes.append("该天气层没有可支撑该方案的共同观测")
    if evidence and not meetable:
        notes.append("乐观值也低于该家庭底线")
    if fragile:
        notes.append("区间跨过底线：不能只凭登记步行时间放行")
    return Assessment(
        plan=plan,
        stratum=stratum,
        counts=counts,
        outcomes=tuple(results),
        estimate=low,
        optimistic=high,
        ci=wilson(counts["completed"], evidence),
        support=support,
        scenario_range=(low, high),
        floor=family.floor,
        meets_floor=meets,
        fragile=fragile,
        notes=tuple(notes),
    )


def evaluate_rain(data: SourceData, family: Family, plan: Plan) -> Assessment:
    return evaluate(data, family, plan, stratum=CONTINGENCY)


def plan_inventory(
    data: SourceData,
    family: Family,
    *,
    usable_rides: Iterable[str] | None = None,
    stratum: str = BASELINE,
) -> tuple[Assessment, ...]:
    """Assess every candidate plan, strongest first."""
    assessments = [
        evaluate(data, family, plan, stratum=stratum)
        for plan in candidate_plans(data, family, usable_rides=usable_rides)
    ]
    assessments.sort(
        key=lambda a: (
            a.meets_floor,
            # A seat the family never boards in any observed trip is not a
            # guarantee and only consumes capacity someone else may need, so
            # avoid it before comparing the completion figures.
            -len(a.phantom_rides()),
            a.estimate,
            a.optimistic,
            a.evidence,
        ),
        reverse=True,
    )
    return tuple(assessments)


def best_plan(
    data: SourceData,
    family: Family,
    *,
    usable_rides: Iterable[str] | None = None,
) -> Assessment | None:
    inventory = plan_inventory(data, family, usable_rides=usable_rides)
    return inventory[0] if inventory else None
