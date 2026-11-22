import csv
from pathlib import Path
import unittest

DATA = Path(__file__).resolve().parents[1] / "data"

TABLES = {
    'slots': ('slot venue service_date starts ends last_checkin spare', 'slot'),
    'rides': ('ride service_date corridor leaves arrives capacity accessible', 'ride'),
    'walks': ('corridor minutes allowed_group', 'corridor allowed_group'),
    'family_route': ('family people trip floor access move_first move_second move_third', 'family'),
    'reserves': ('reservation family ride seats state issued_at', 'reservation'),
    'activation': ('trigger group scope deadline applies_to facts', 'trigger'),
    'notices': ('notice ride state effective_at received_at', 'notice'),
    'observations': ('record trip day corridor segment weather demand scheduled_minutes elapsed_minutes outcome mode', 'record'),
}


class SourceDataTests(unittest.TestCase):
    def load(self, name):
        fields, _ = TABLES[name]
        with (DATA / (name + ".csv")).open(encoding="utf-8", newline="") as handle:
            contents = csv.DictReader(handle)
            self.assertEqual(contents.fieldnames, fields.split())
            records = list(contents)
        self.assertTrue(records)
        self.assertTrue(all(None not in row and None not in row.values() for row in records))
        return records

    def test_format_and_record_keys(self):
        for name, (_, key) in TABLES.items():
            with self.subTest(table=name):
                records = self.load(name)
                values = [tuple(row[field] for field in key.split()) for row in records]
                self.assertTrue(all(all(value) for value in values))
                self.assertEqual(len(values), len(set(values)))

    def test_related_source_records(self):
        rows = {name: self.load(name) for name in TABLES}
        logs = rows["observations"]
        groups = {(r["day"], r["corridor"]) for r in logs}
        self.assertLess(len(groups), len(logs))
        self.assertTrue(any(r["outcome"] == "incomplete" for r in logs))
        self.assertEqual({"exclusive", "shared"}, {r["scope"] for r in rows["activation"]})
        self.assertTrue({r["ride"] for r in rows["notices"]} <= {r["ride"] for r in rows["rides"]})
        self.assertTrue(any(r["state"] == "locked" for r in rows["reserves"]))
