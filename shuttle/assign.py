"""Joint assignment: choose one plan per family under floors and seat limits.

Individual recommendations do not add up.  Give every family its own best plan
and the shared shuttles overflow; a family rescued only by a seat can lose it
to a family the operator moved onto the same ride.  Choosing plans one at a
time hides that, so this module chooses them together.

The search is exact over the small candidate sets in this data.  It ranks
assignments by, in order:

1. families that clear their own floor - the sold promise to the customer;
2. total head-count-weighted margin over those floors, so a large party is not
   traded away for a small one without reason;
3. fewer families whose success depends on a walk replacing a missed ride (a
   fallback is a bonus, not a guarantee);
4. fewer promises on *running* rides left unused, because moving a family off
   something it was told it holds costs a phone call and possibly a refusal;
5. fewer shuttle seats held.

Promises on a ride a notice has already stopped are a different case: nobody
can keep them, so they neither veto a plan nor count as a broken promise here.
They are reported to the officer as stranded and handled by ``duty``.

Every plan is evaluated against the rides running at the reference moment, so a
cancelled ride can appear in the report but can never be silently re-used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Mapping

from . import risk, seats
from .model import SourceData, notice_state
from .risk import Assessment, Plan


@dataclass(frozen=True)
class Choice:
    family: str
    assessment: Assessment
    keeps_running_promises: bool

    @property
    def plan(self) -> Plan:
        return self.assessment.plan

    @property
    def meets_floor(self) -> bool:
        return self.assessment.meets_floor


@dataclass(frozen=True)
class JointPlan:
    at: datetime
    choices: Mapping[str, Choice]
    loads: Mapping[str, seats.RideLoad]
    conflicts: tuple[seats.RideLoad, ...]
    score: tuple[int, ...]
    unmet: tuple[str, ...]
    unused_promises: tuple[str, ...]
    stranded_promises: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def plans(self) -> dict[str, Plan]:
        return {family_id: choice.plan for family_id, choice in self.choices.items()}

    def describe(self) -> str:
        lines = []
        for family_id, choice in sorted(self.choices.items()):
            a = choice.assessment
            status = "达标" if a.meets_floor else "未达底线"
            confirm = "" if choice.keeps_running_promises else "（需向家庭确认席位变动）"
            lines.append(
                f"{family_id} {a.plan.label()} 完成率 {a.estimate:.2f}"
                f"（{a.scenario_range[0]:.2f}-{a.scenario_range[1]:.2f}，n={a.evidence}）"
                f"{status}{confirm}"
            )
        return "\n".join(lines)


def inventory(
    data: SourceData,
    at: datetime,
    *,
    stratum: str = risk.BASELINE,
) -> dict[str, tuple[Assessment, ...]]:
    """Every candidate assessment per family, against the rides running at ``at``."""
    usable = seats.running(data, at)
    return {
        family_id: tuple(
            risk.evaluate(data, family, plan, stratum=stratum)
            for plan in risk.candidate_plans(data, family, usable_rides=usable)
        )
        for family_id, family in data.families.items()
    }


def unused_running_promises(
    data: SourceData,
    family: str,
    plan: Plan,
    at: datetime,
) -> tuple[str, ...]:
    """Promised rides still running that this plan does not use."""
    out = []
    for claim in seats.locked_claims(data, at=at, include_stopped=True):
        if claim.family != family or claim.ride in plan.bus_rides():
            continue
        if notice_state(data, claim.ride, at) == "running":
            out.append(claim.ride)
    return tuple(out)


def _seat_usage(data: SourceData, plans: Mapping[str, Plan]) -> int:
    return sum(
        data.families[family_id].people
        for family_id, plan in plans.items()
        for _ in plan.bus_rides()
    )


def _score(data: SourceData, choices: Mapping[str, Choice]) -> tuple[int, ...]:
    met = sum(1 for c in choices.values() if c.meets_floor)
    margin = sum(
        (c.assessment.estimate - c.assessment.floor) * data.families[fid].people
        for fid, c in choices.items()
        if c.meets_floor
    )
    fallback_dependent = sum(
        1 for c in choices.values() if c.meets_floor and c.assessment.fallback_completions
    )
    unkept = sum(1 for c in choices.values() if not c.keeps_running_promises)
    plans = {f: c.plan for f, c in choices.items()}
    return (met, round(margin, 6), -fallback_dependent, -unkept, -_seat_usage(data, plans))


def solve(
    data: SourceData,
    at: datetime,
    *,
    stratum: str = risk.BASELINE,
    max_per_family: int = 8,
) -> JointPlan:
    """Exact best joint assignment over the candidate plans of this day.

    Candidate sets are capped per family, keeping the plans that are strongest
    on floor, estimate, and seat economy, so the cross-product stays small while
    no family loses its best options.
    """
    available = inventory(data, at, stratum=stratum)
    options: list[tuple[Choice, ...]] = []
    for family_id in sorted(available):
        ranked = sorted(available[family_id], key=_rank, reverse=True)[:max_per_family]
        choices = tuple(
            Choice(
                family_id,
                assessment,
                keeps_running_promises=not unused_running_promises(data, family_id, assessment.plan, at),
            )
            for assessment in ranked
        )
        if not choices:  # pragma: no cover - every family has at least one plan
            raise ValueError(f"{family_id} 没有任何候选方案")
        options.append(choices)

    best: tuple[tuple[int, ...], tuple[Choice, ...]] | None = None
    fallback: tuple[tuple[int, ...], tuple[Choice, ...]] | None = None
    for combination in product(*options):
        plans = {c.family: c.plan for c in combination}
        ok, _ = seats.feasible(data, plans, at=at)
        choice_map = {c.family: c for c in combination}
        score = _score(data, choice_map)
        if ok:
            if best is None or score > best[0]:
                best = (score, combination)
        elif fallback is None or score > fallback[0]:
            fallback = (score, combination)

    if best is None:
        assert fallback is not None
        score, combination = fallback
        notes = ("现有席位下没有无冲突的整组方案，下面是代价最小的冲突方案",)
    else:
        score, combination = best
        notes = ()

    choices = {c.family: c for c in combination}
    plans = {family_id: choice.plan for family_id, choice in choices.items()}
    loads = seats.loads(data, plans, at=at)
    conflicts = tuple(load for load in loads.values() if load.shortfall)
    unmet = tuple(sorted(fid for fid, c in choices.items() if not c.meets_floor))
    unused = tuple(
        f"{claim.reservation}({claim.family}·{claim.ride})"
        for claim in seats.unused_promises(data, plans, at=at)
        if notice_state(data, claim.ride, at) == "running"
    )
    stranded = tuple(
        f"{claim.reservation}({claim.family}·{claim.ride})"
        for claim in seats.unused_promises(data, plans, at=at)
        if notice_state(data, claim.ride, at) != "running"
    )
    return JointPlan(at, choices, loads, conflicts, score, unmet, unused, stranded, notes)


def _rank(assessment: Assessment) -> tuple[int, float, int, int]:
    return (
        int(assessment.meets_floor),
        assessment.estimate,
        -len(assessment.phantom_rides()),
        int(assessment.fallback_completions == 0),
    )


def explain_difference(left: JointPlan, right: JointPlan) -> tuple[str, ...]:
    """Human-readable differences between two joint plans."""
    out = []
    for family_id in sorted(set(left.choices) | set(right.choices)):
        a = left.choices.get(family_id)
        b = right.choices.get(family_id)
        if a == b:
            continue
        left_text = f"{a.plan.label()}（{a.assessment.estimate:.2f}）" if a else "无"
        right_text = f"{b.plan.label()}（{b.assessment.estimate:.2f}）" if b else "无"
        if a and b and a.plan.slots == b.plan.slots:
            kind = "改接驳"
        elif a and b and a.plan.bus_rides() == b.plan.bus_rides():
            kind = "改场次"
        else:
            kind = "改场次与接驳"
        out.append(f"{family_id}：{kind} {left_text} → {right_text}")
    for load in right.conflicts:
        out.append(load.describe())
    return tuple(out)
