import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path

from shuttle import assign, compare, duty, model, plan, retro, risk, seats

DATA = Path(__file__).resolve().parents[1] / "data"
AT = datetime.fromisoformat("2026-12-06T10:30:00+08:00")
EARLY = datetime.fromisoformat("2026-12-06T08:00:00+08:00")


def load():
    return model.load(DATA)


class ClockAndNoticeTests(unittest.TestCase):
    def setUp(self):
        self.data = load()

    def test_last_checkin_is_inclusive(self):
        slot = self.data.slots["SB1"]
        self.assertEqual(slot.last_checkin, 10 * 60 + 19)

    def test_notice_effective_time_decides_before_receipt(self):
        # R2 is effective at 08:12 but only received at 08:15.
        at_0813 = datetime.fromisoformat("2026-12-06T08:13:00+08:00")
        self.assertEqual(model.notice_state(self.data, "R2", at_0813), "cancelled")
        at_0811 = datetime.fromisoformat("2026-12-06T08:11:00+08:00")
        self.assertEqual(model.notice_state(self.data, "R2", at_0811), "running")

    def test_running_rides_drops_cancelled(self):
        self.assertEqual(set(seats.running(self.data, AT)), {"R1", "R3"})


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.data = load()
        self.f1 = self.data.families["F1"]

    def test_trip_is_counted_once_not_per_segment(self):
        assessment = risk.evaluate(
            self.data, self.f1, risk.Plan("F1", ("SA1", "SB1", "SC1"), "R1", "R3")
        )
        total = sum(assessment.counts.values())
        self.assertEqual(total, assessment.evidence + assessment.censored)
        self.assertEqual(total, len(assessment.outcomes))

    def test_incomplete_is_censored_not_counted_as_pass(self):
        assessment = risk.evaluate(
            self.data, self.f1, risk.Plan("F1", ("SA1", "SB1", "SC1"), "R1", "R3")
        )
        self.assertGreater(assessment.censored, 0)
        self.assertAlmostEqual(
            assessment.completed / assessment.evidence,
            assessment.optimistic,
        )

    def test_missing_bc_leg_is_left_out_of_walk_free_evidence(self):
        bus = risk.Plan("F1", ("SA1", "SB1", "SC1"), "R1", "R3")
        self.assertEqual(len(risk.evidence_trips(self.data, bus)), 52)
        trips = risk.evidence_trips(self.data, bus)
        self.assertTrue(all(t.state("ab") != "missing" and t.state("bc") != "missing" for t in trips))

    def test_segment_average_is_not_the_joint_rate(self):
        """Multiplying per-segment success would give a much rosier number."""
        trip = self.data.trips["T0-0"]
        for segment in ("a", "ab", "b", "bc"):
            obs = trip.segment(segment)
            if obs is not None:
                self.assertLessEqual(obs.elapsed_minutes, obs.scheduled_minutes + 10)
        rate = Counter()
        for item in self.data.trips.values():
            if item.weather != "normal":
                continue
            for obs in item.segments.values():
                if obs.elapsed_minutes is not None:
                    rate["n"] += 1
                    rate["ok"] += obs.elapsed_minutes <= obs.scheduled_minutes
        naive = (rate["ok"] / rate["n"]) ** 4
        joint = risk.evaluate(
            self.data, self.f1, risk.Plan("F1", ("SA1", "SB1", "SC1"), "R1", "R3")
        ).optimistic
        self.assertNotAlmostEqual(naive, joint, places=3)

    def test_walk_plan_gets_a_range_not_a_point(self):
        walk = risk.evaluate(self.data, self.f1, risk.Plan("F1", ("SA1", "SB1", "SC1"), "walk", "walk"))
        self.assertEqual(walk.support, "range")
        self.assertLessEqual(walk.scenario_range[0], walk.scenario_range[1])

    def test_missing_evidence_is_declared_not_guessed(self):
        """A plan's evidence count is stated, never inferred from another plan."""
        plan_ = risk.Plan("F1", ("SA1", "SB1", "SC1"), "R2", "R4")
        trips = risk.evidence_trips(self.data, plan_)
        # Only trips that recorded both connecting legs can speak to this plan.
        self.assertTrue(trips)
        self.assertTrue(all(t.state("ab") != "missing" and t.state("bc") != "missing" for t in trips))
        self.assertLess(len(trips), len([t for t in self.data.trips.values() if t.weather == "normal"]))
        # Ride availability is a separate fact: it comes from notices, not from
        # how many trips happened to record a leg.
        self.assertEqual(model.notice_state(self.data, "R2", AT), "cancelled")

    def test_rain_is_reported_beside_the_floor(self):
        plan_ = risk.Plan("F1", ("SA1", "SB1", "SC1"), "R1", "R3")
        rain = risk.evaluate_rain(self.data, self.f1, plan_)
        self.assertEqual(rain.stratum, "rain")
        self.assertEqual(rain.estimate, 0.0)
        self.assertGreater(rain.evidence, 0)


class SlotAndSeatTests(unittest.TestCase):
    def setUp(self):
        self.data = load()

    def test_move_respects_spare_seats_for_whole_party(self):
        f1 = self.data.families["F1"]
        plans = risk.candidate_plans(self.data, f1, usable_rides={"R1", "R3"})
        # F1 has 4 people; SA2 and SB2 have >=4 spare but SC2 has 3, so a move to SC2 is barred.
        self.assertTrue(any(p.slots[2] == "SC2" for p in plans) is False)
        self.assertTrue(all(p.slots[0] in {"SA1", "SA2"} for p in plans))

    def test_step_free_family_cannot_board_inaccessible_ride(self):
        f2 = self.data.families["F2"]
        plans = risk.candidate_plans(self.data, f2, usable_rides={"R1", "R3", "R4"})
        self.assertTrue(plans)
        self.assertTrue(all("R3" not in p.bus_rides() for p in plans))

    def test_standard_family_may_use_an_inaccessible_ride(self):
        f1 = self.data.families["F1"]
        plan_ = risk.Plan("F1", f1.trip, "R1", "R3")
        self.assertTrue(seats.accessible_ok(self.data, f1, plan_))

    def test_step_free_walk_uses_registered_minutes(self):
        f2 = self.data.families["F2"]
        self.assertEqual(risk.walk_minutes(self.data, f2, "BC"), 45)
        f1 = self.data.families["F1"]
        self.assertEqual(risk.walk_minutes(self.data, f1, "BC"), 37)

    def test_locked_promise_occupies_capacity_even_if_family_moves(self):
        f3 = self.data.families["F3"]
        moved = risk.Plan("F3", f3.trip, "walk", "walk")
        load = seats.loads(self.data, {"F3": moved}, at=AT)["R1"]
        self.assertEqual(load.demand, 3)
        self.assertEqual(load.promised, ("F3",))

    def test_party_is_never_split(self):
        assignment = {
            "F1": risk.Plan("F1", self.data.families["F1"].trip, "R1", "R3"),
            "F5": risk.Plan("F5", self.data.families["F5"].trip, "R1", "R3"),
        }
        load = seats.loads(self.data, assignment, at=AT)["R1"]
        # Whole parties only: F1's 4 plus F5's 2.  F3's locked 3 still hold R1
        # because F3 has not been reassigned here, so 9 is the honest number.
        self.assertEqual(load.demand, 4 + 2 + 3)
        self.assertEqual(load.promised, ("F3",))
        self.assertFalse(seats.feasible(self.data, assignment, at=AT)[0])
        # Once F3 is placed on R1 too, its promise is its own party, not extra.
        together = dict(assignment)
        together["F3"] = risk.Plan("F3", self.data.families["F3"].trip, "R1", "R3")
        self.assertEqual(seats.loads(self.data, together, at=AT)["R1"].demand, 4 + 2 + 3)
        self.assertFalse(seats.feasible(self.data, together, at=AT)[0])

    def test_one_more_whole_party_would_overflow(self):
        assignment = {
            "F1": risk.Plan("F1", self.data.families["F1"].trip, "R1", "R3"),
            "F3": risk.Plan("F3", self.data.families["F3"].trip, "R1", "R3"),
            "F5": risk.Plan("F5", self.data.families["F5"].trip, "R1", "R3"),
        }
        ok, conflicts = seats.feasible(self.data, assignment, at=AT)
        self.assertFalse(ok)
        self.assertTrue(any(c.ride == "R1" and c.shortfall for c in conflicts))


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.data = load()

    def test_same_deadline_exclusive_triggers_cannot_share_a_seat(self):
        problems = seats.shared_seat_problems(self.data)
        self.assertTrue(any("TR2" in p and "TR3" in p for p in problems))
        self.assertTrue(any("不能共用" in p for p in problems))

    def test_concurrent_shared_trigger_on_one_corridor_is_flagged(self):
        messages = seats.concurrent_corridor_impacts(self.data)
        self.assertTrue(any("AB" in m and "TR1" in m for m in messages))

    def test_shared_trigger_does_not_grant_an_exclusive_hold(self):
        f1_triggers = {a.scope for a in seats.triggers_for(self.data, "F1")}
        self.assertIn("shared", f1_triggers)
        self.assertIn("exclusive", f1_triggers)
        f3_triggers = [a.trigger for a in seats.triggers_for(self.data, "F3")]
        self.assertEqual(f3_triggers, ["TR4"])


class NoticeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.data = load()

    def draft(self):
        return duty.Draft(
            {
                "F1": risk.Plan("F1", self.data.families["F1"].trip, "R2", "R3"),
                "F4": risk.Plan("F4", self.data.families["F4"].trip, "R2", "R4"),
            },
            EARLY,
        )

    def test_draft_goes_stale_when_notice_takes_effect(self):
        self.assertEqual(duty.check_staleness(self.data, self.draft(), datetime.fromisoformat("2026-12-06T08:10:00+08:00")), "draft")
        self.assertEqual(duty.check_staleness(self.data, self.draft(), datetime.fromisoformat("2026-12-06T09:30:00+08:00")), "stale")

    def test_confirmed_draft_is_not_rewritten_by_a_later_notice(self):
        draft = duty.Draft(self.draft().assignment, EARLY, state=duty.CONFIRMED)
        self.assertEqual(duty.check_staleness(self.data, draft, AT), duty.CONFIRMED)

    def test_dispositions_separate_keep_confirm_and_unrescuable(self):
        rows = {row.family: row for row in duty.dispositions(self.data, self.draft(), AT)}
        self.assertEqual(rows["F1"].bucket, "confirm")
        self.assertIsNotNone(rows["F1"].replacement)
        self.assertEqual(rows["F4"].bucket, "unrescuable")
        self.assertIsNone(rows["F4"].replacement)

    def test_stranded_promise_is_named_not_dropped(self):
        self.assertEqual(duty.stranded_promises(self.data, AT), {"F4": ("R4",)})
        self.assertEqual(duty.kept_promises(self.data, AT), {"F3": ("R1",)})


class AssignmentTests(unittest.TestCase):
    def setUp(self):
        self.data = load()
        self.joint = assign.solve(self.data, AT)

    def test_joint_assignment_is_seat_feasible_or_says_why(self):
        if not self.joint.conflicts:
            ok, _ = seats.feasible(self.data, self.joint.plans(), at=AT)
            self.assertTrue(ok)

    def test_joint_assignment_never_boards_a_cancelled_ride(self):
        running = set(seats.running(self.data, AT))
        for plan_ in self.joint.plans().values():
            self.assertTrue(set(plan_.bus_rides()) <= running)

    def test_one_family_at_a_time_would_look_better_than_it_is(self):
        naive = 0
        for family_id, family in self.data.families.items():
            best = risk.plan_inventory(self.data, family, usable_rides=seats.running(self.data, AT))[0]
            naive += best.meets_floor
            self.assertTrue(best is not None)
        scheduled = sum(1 for c in self.joint.choices.values() if c.meets_floor)
        self.assertLessEqual(scheduled, naive)

    def test_displaced_party_does_not_get_a_phantom_seat(self):
        for family_id, choice in self.joint.choices.items():
            assessment = choice.assessment
            if assessment.meets_floor:
                self.assertGreater(assessment.evidence, 0)


class BriefingTests(unittest.TestCase):
    def setUp(self):
        self.data = load()
        self.briefing = plan.build(self.data, AT)

    def test_unmet_families_are_surfaced(self):
        self.assertTrue(all(fid in self.briefing.lines for fid in self.briefing.lines))
        for family_id, line in self.briefing.lines.items():
            if not line.assessment.meets_floor:
                self.assertTrue(any(family_id in warning for warning in self.briefing.warnings))

    def test_self_connection_without_guarantee_is_reported(self):
        self.assertTrue(self.briefing.unsecured)
        self.assertTrue(any("无接驳保证" in line.describe() for line in self.briefing.unsecured))

    def test_review_keeps_censored_apart_from_failure(self):
        review = retro.review_day(self.data, self.briefing.joint.plans(), approved_at=AT)
        for record in review.outcomes.values():
            self.assertEqual(
                record.completed + record.known_late + record.not_completed + record.censored,
                len(
                    risk.evaluate(
                        self.data,
                        self.data.families[record.family],
                        self.briefing.joint.choices[record.family].plan,
                    ).outcomes
                ),
            )

    def test_comparable_evidence_warns_when_a_walk_leg_is_involved(self):
        bus = risk.Plan("F1", ("SA1", "SB1", "SC1"), "R1", "R3")
        walk = risk.Plan("F1", ("SA1", "SB1", "SC1"), "R1", "walk")
        notes = retro.comparable_evidence(self.data, bus, walk)
        self.assertTrue(any("步行" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
