"""The briefing for one planning moment.

This is the read-only view the officer works from.  It joins three things that
must agree with one another:

* the joint assignment (``assign``), so the seat count is already net of shared
  rides and kept promises;
* each family's own evidence (``risk``), so the completion figure quoted is the
  one behind the chosen plan, not the best figure any plan could reach;
* the notice and promise state at that moment (``duty`` / ``seats``).

Nothing is mutated.  If the briefing cannot be assembled because no plan keeps
a promise, that is raised rather than papered over.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from . import assign, compare, duty, risk, seats
from .model import Family, SourceData
from .risk import Assessment, Plan


@dataclass(frozen=True)
class FamilyLine:
    family: Family
    plan: Plan
    assessment: Assessment
    held_rides: tuple[str, ...]
    kept_promises: tuple[str, ...]
    recovery_blocked: tuple[str, ...]
    rain: Assessment | None
    notes: tuple[str, ...]

    def describe(self) -> str:
        a = self.assessment
        low, high = a.scenario_range
        held = "、".join(self.held_rides) or "无"
        status = "达标" if a.meets_floor else "未达底线"
        bits = [
            f"{self.family.family}（{self.family.people} 人，{self.family.access}，底线 {self.family.floor:.2f}）",
            f"现持 {held}",
            f"安排 {a.plan.label()}",
            f"完成率 {a.estimate:.2f}（支持区间 {low:.2f}-{high:.2f}，n={a.evidence}，{a.support}）",
            status,
        ]
        if self.rain is not None:
            bits.append(f"雨天参照 {self.rain.estimate:.2f}")
        return "；".join(bits)


@dataclass(frozen=True)
class Briefing:
    at: datetime
    joint: assign.JointPlan
    lines: Mapping[str, FamilyLine]
    warnings: tuple[str, ...]
    notices: tuple[duty.NoticeImpact, ...]
    stranded: Mapping[str, tuple[str, ...]]
    unsecured: tuple[duty.SelfConnect, ...]

    def describe(self) -> str:
        out = [f"计划时点 {self.at:%Y-%m-%d %H:%M}"]
        out.extend(line.describe() for line in self.lines.values())
        out.append("席位：")
        for load in self.joint.loads.values():
            held = "、".join(load.families) or "无人"
            promised = f"，另有已承诺 {('、'.join(load.promised))}" if load.promised else ""
            out.append(f"  {load.ride} {load.demand}/{load.capacity} 席（{held}{promised}）")
        if self.notices:
            out.append("通知：")
            out.extend("  " + notice.describe() for notice in self.notices)
        if self.warnings:
            out.append("注意：")
            out.extend("  " + warning for warning in self.warnings)
        return "\n".join(out)


def build(data: SourceData, at: datetime, *, stratum: str = risk.BASELINE) -> Briefing:
    """Assemble the briefing for ``at`` from a joint assignment."""
    joint = assign.solve(data, at, stratum=stratum)
    lines: dict[str, FamilyLine] = {}
    for family_id, choice in joint.choices.items():
        family = data.families[family_id]
        held = tuple(
            claim.ride
            for claim in seats.locked_claims(data, at=at, include_stopped=True)
            if claim.family == family_id
        )
        kept = tuple(claim.reservation for claim in seats.locked_claims(data, at=at) if claim.family == family_id)
        alternatives = {family_id: tuple(a.plan for a in _inventory(data, family, at))}
        blocked = seats.recovery_pressure(data, joint.plans(), alternatives, at=at).get(family_id, ())
        rain = risk.evaluate_rain(data, family, choice.plan)
        notes = list(choice.assessment.notes)
        if choice.assessment.phantom_rides():
            notes.append(f"标了 {'、'.join(choice.assessment.phantom_rides())}，观测里从未用上，不能算作保证")
        if choice.assessment.fallback_completions:
            notes.append(f"{choice.assessment.fallback_completions} 趟靠步行补位才按时，席位不是唯一保障")
        lines[family_id] = FamilyLine(
            family, choice.plan, choice.assessment, held, kept, blocked, rain, tuple(notes)
        )

    warnings = list(seats.shared_seat_problems(data)) + list(seats.concurrent_corridor_impacts(data))
    for family_id, line in lines.items():
        if not line.assessment.meets_floor:
            warnings.append(
                f"{family_id} 在 {at:%H:%M} 可用的班次下无法达到底线 "
                f"{line.family.floor:.2f}（最好 {line.assessment.optimistic:.2f}），需值班员决断"
            )
    for load in joint.conflicts:
        warnings.append(load.describe())

    notice_view = duty.notice_impacts(data, at)
    stranded = duty.stranded_promises(data, at)
    assignment = joint.plans()
    unsecured = duty.self_connects(data, assignment, at)
    return Briefing(at, joint, lines, tuple(dict.fromkeys(warnings)), notice_view, stranded, unsecured)


def _inventory(data: SourceData, family: Family, at: datetime) -> tuple[Assessment, ...]:
    return risk.plan_inventory(data, family, usable_rides=seats.running(data, at))


def slot_versus_seat(
    data: SourceData,
    family: Family,
    *,
    at: datetime,
) -> dict[str, tuple[compare.Option, ...]]:
    """Options split into the two operator levers, for one family."""
    options = compare.compare(data, family, at=at)
    out: dict[str, list[compare.Option]] = {"slot": [], "seat": []}
    for option in options:
        out[option.kind].append(option)
    return {kind: tuple(sorted(items, key=lambda o: (o.feasible, o.assessment.estimate), reverse=True)) for kind, items in out.items()}
