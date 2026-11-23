"""Post-run review: what was approved, on what evidence, and how it turned out.

The review has two halves that must not be confused.

Approval record
    The evidence behind an approved arrangement: which historical trips
    supported it, the support range, the floor, and the seat commitments that
    were made.  This is reconstructable after the fact and is what an audit
    reads.

Outcome record
    What actually happened.  The contract wants known lateness, a journey that
    never finished, and a genuine completion kept apart.  A trip that was never
    observed to finish is not a completion and is not a known failure either; it
    carries its own bucket.

The two are joined per family so next week's similar line can see how a given
support figure behaved in practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from . import risk, seats
from .model import Family, SourceData
from .risk import Assessment, Plan, TripResult


@dataclass(frozen=True)
class ApprovalRecord:
    family: str
    plan: Plan
    floor: float
    estimate: float
    support: str
    scenario_range: tuple[float, float]
    evidence: tuple[str, ...]
    causes: Mapping[str, int]
    seat_commitments: tuple[str, ...]
    approved_at: datetime

    def describe(self) -> str:
        low, high = self.scenario_range
        seats_text = "、".join(self.seat_commitments) or "无"
        return (
            f"{self.family} 批准 {self.plan.label()}：完成率 {self.estimate:.2f}"
            f"（区间 {low:.2f}-{high:.2f}，{self.support}，n={len(self.evidence)}），"
            f"底线 {self.floor:.2f}；席位承诺 {seats_text}"
        )


@dataclass(frozen=True)
class OutcomeRecord:
    family: str
    completed: int
    known_late: int
    not_completed: int
    censored: int
    examples: tuple[str, ...]

    def describe(self) -> str:
        bits = [f"完成 {self.completed}", f"确知迟到 {self.known_late}", f"旅程未走完 {self.not_completed}"]
        if self.censored:
            bits.append(f"观测未结束 {self.censored}（不作完成也不作失败）")
        return f"{self.family}：" + "，".join(bits)


@dataclass(frozen=True)
class Review:
    approved_at: datetime
    approvals: Mapping[str, ApprovalRecord]
    outcomes: Mapping[str, OutcomeRecord]
    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        return "\n".join(
            [record.describe() for record in self.approvals.values()]
            + [record.describe() for record in self.outcomes.values()]
        )


def approval_record(
    data: SourceData,
    family: Family,
    plan: Plan,
    *,
    approved_at: datetime,
    stratum: str = risk.BASELINE,
) -> ApprovalRecord:
    assessment = risk.evaluate(data, family, plan, stratum=stratum)
    evidence = tuple(
        f"{result.trip}({result.day},{result.weather})"
        for result in assessment.outcomes
        if result.outcome != "incomplete"
    )
    commitments = tuple(
        f"{claim.reservation}:{claim.ride}"
        for claim in seats.locked_claims(data, include_stopped=True)
        if claim.family == family.family
    )
    return ApprovalRecord(
        family=family.family,
        plan=plan,
        floor=family.floor,
        estimate=assessment.estimate,
        support=assessment.support,
        scenario_range=assessment.scenario_range,
        evidence=evidence,
        causes=assessment.causes(),
        seat_commitments=commitments,
        approved_at=approved_at,
    )


def outcome_record(family: Family, assessment: Assessment) -> OutcomeRecord:
    counts = assessment.counts
    # Censored observations are cited separately: they are not evidence of a
    # completion and not evidence of a known failure either.
    examples = tuple(
        result.describe()
        for result in assessment.outcomes
        if result.outcome in {"known_late", "not_completed"}
    )[:5]
    return OutcomeRecord(
        family=family.family,
        completed=counts["completed"],
        known_late=counts["known_late"],
        not_completed=counts["not_completed"],
        censored=counts["incomplete"],
        examples=examples,
    )


def review_day(
    data: SourceData,
    approved: Mapping[str, Plan],
    *,
    approved_at: datetime,
    stratum: str = risk.BASELINE,
) -> Review:
    """Pair each approved arrangement with its observed outcome."""
    approvals: dict[str, ApprovalRecord] = {}
    outcomes: dict[str, OutcomeRecord] = {}
    notes: list[str] = []
    for family_id, plan in sorted(approved.items()):
        family = data.families[family_id]
        approvals[family_id] = approval_record(data, family, plan, approved_at=approved_at, stratum=stratum)
        assessment = risk.evaluate(data, family, plan, stratum=stratum)
        outcomes[family_id] = outcome_record(family, assessment)
        rain = risk.evaluate_rain(data, family, plan)
        if rain.estimate == 0.0 and rain.evidence:
            notes.append(f"{family_id} 的方案在 rain 观测层完成率为 0，雨天须依赖触发预案而非按基准放行")
    return Review(approved_at, approvals, outcomes, tuple(notes))


def comparable_evidence(
    data: SourceData,
    plan_a: Plan,
    plan_b: Plan,
    *,
    stratum: str = risk.BASELINE,
) -> tuple[str, ...]:
    """Say plainly whether two plans rest on the same joint observation.

    Two plans are comparable only when their evidence sets overlap on each
    corridor; a plan that walks a corridor has no observation for that leg and
    therefore cannot be compared to a bus plan corridor-for-corridor.
    """
    trips_a = {trip.trip for trip in risk.evidence_trips(data, plan_a, stratum)}
    trips_b = {trip.trip for trip in risk.evidence_trips(data, plan_b, stratum)}
    shared = trips_a & trips_b
    notes = [f"{plan_a.label()} 与 {plan_b.label()} 共有 {len(shared)} 个同层观测趟"]
    if plan_a.has_walk() ^ plan_b.has_walk():
        notes.append("一侧含步行段，无用时观测，另一侧不能与其逐段比")
    if not shared:
        notes.append("没有共同观测，两个方案之间只能各给区间，不能直接排先后")
    return tuple(notes)
