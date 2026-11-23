"""Command-line briefing for the duty officer.

    python3 -m shuttle brief --at 2026-12-06T10:30:00+08:00
    python3 -m shuttle compare F1 --at 2026-12-06T10:30:00+08:00
    python3 -m shuttle duty --at 2026-12-06T10:30:00+08:00
    python3 -m shuttle review --at 2026-12-06T10:30:00+08:00

Every command reads ``data/`` and prints; none of them rewrites a source record.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from . import assign, compare, duty, model, plan, retro, risk, seats


def _moment(text: str) -> datetime:
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"不是 ISO 时刻: {text!r}")


def cmd_brief(args: argparse.Namespace) -> int:
    data = model.load(args.data)
    briefing = plan.build(data, args.at)
    print(briefing.describe())
    if briefing.stranded:
        print("已停开线路上的承诺（须值班员另行处理）：")
        for family, rides in briefing.stranded.items():
            print(f"  {family}: {'、'.join(rides)}")
    if briefing.unsecured:
        print("自行换乘、未取得接驳保证：")
        for line in briefing.unsecured:
            print("  " + line.describe())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    data = model.load(args.data)
    family = data.families[args.family]
    branches = plan.slot_versus_seat(data, family, at=args.at)
    current = compare.current_plan(data, family)
    print(f"{family.family}（{family.people} 人，底线 {family.floor:.2f}，{family.access}）")
    print(f"现状：{current.label()}")
    best_overall = None
    for kind, title in (("slot", "改场次"), ("seat", "留接驳席")):
        print(f"\n{title}：")
        options = branches[kind]
        if not options:
            print("  没有可选方案")
            continue
        shown = [o for o in options if o.feasible] or options[:1]
        for option in shown:
            print("  " + option.describe())
            if option.cohort:
                print(f"    同批家庭：{'、'.join(option.cohort)}")
            if option.blocked_recovery:
                print(f"    会挤占的恢复席位：{'、'.join(option.blocked_recovery)}")
            for note in option.notes:
                print(f"    · {note}")
            if best_overall is None or (
                option.feasible, option.assessment.estimate
            ) > (
                best_overall.feasible, best_overall.assessment.estimate
            ):
                best_overall = option
    if best_overall is not None:
        print("\n失败原因（按段、班次与签到窗口）：")
        reasons = compare.failure_reasons(data, family, best_overall.plan, at=args.at)
        if not reasons:
            print("  该方案在本日观测中没有可指认的失败段")
        for reason in reasons:
            print("  " + reason)
    return 0


def cmd_duty(args: argparse.Namespace) -> int:
    data = model.load(args.data)
    if args.draft_family:
        assignment = {}
        for spec in args.draft_family:
            family_id, _, rides = spec.partition("=")
            family = data.families[family_id]
            ab, _, bc = rides.partition(":")
            assignment[family_id] = risk.Plan(family_id, family.trip, ab or risk.WALK, bc or risk.WALK)
    else:
        assignment = assign.solve(data, args.at).plans()
    draft = duty.Draft(assignment, args.prepared_at or args.at)
    state = duty.check_staleness(data, draft, args.at)
    print(f"运行单状态：{state}")
    for impact in duty.notice_impacts(data, args.at, draft):
        print("  " + impact.describe())
    print("家庭处置：")
    for disposition in duty.dispositions(data, draft, args.at):
        print(f"  [{disposition.bucket}] " + disposition.describe())
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    data = model.load(args.data)
    briefing = plan.build(data, args.at)
    review = retro.review_day(data, briefing.joint.plans(), approved_at=args.at)
    print("批准依据：")
    for record in review.approvals.values():
        print("  " + record.describe())
    print("\n运行结果：")
    for record in review.outcomes.values():
        print("  " + record.describe())
        for example in record.examples:
            print(f"    · {example}")
    for note in review.notes:
        print("  · " + note)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shuttle", description="跨馆短线赶场风险与转场席位")
    parser.add_argument("--data", default=None, help="data 目录，默认仓库内 data/")
    sub = parser.add_subparsers(dest="command", required=True)

    brief = sub.add_parser("brief", help="当前时点的整组安排与座位占用")
    brief.add_argument("--at", type=_moment, required=True)
    brief.set_defaults(func=cmd_brief)

    compare_cmd = sub.add_parser("compare", help="比较改场次与留接驳席")
    compare_cmd.add_argument("family")
    compare_cmd.add_argument("--at", type=_moment, required=True)
    compare_cmd.set_defaults(func=cmd_compare)

    duty_cmd = sub.add_parser("duty", help="通知生效后的处置分桶")
    duty_cmd.add_argument("--at", type=_moment, required=True)
    duty_cmd.add_argument("--prepared-at", type=_moment, default=None)
    duty_cmd.add_argument(
        "--draft-family",
        action="append",
        help="草稿家庭，形如 F1=R2:R3 或 F1=walk:walk",
    )
    duty_cmd.set_defaults(func=cmd_duty)

    review = sub.add_parser("review", help="复盘：批准依据与运行结果")
    review.add_argument("--at", type=_moment, required=True)
    review.set_defaults(func=cmd_review)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (model.SourceError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
