#!/usr/bin/env python3
"""Utilities for Insta360 sidecar compatibility matrix workflow.

Commands:
  generate  Create matrix rows for all combinations of test dimensions.
  summarize Compute pass-rate and decision-gate recommendation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DEFAULT_SIDECAR_TYPES = [
    "studio_generated",
    "ours_generated",
    "ours_mutated_from_studio",
]
CSV_HEADERS = [
    "date",
    "tester",
    "camera_model",
    "studio_version",
    "clip_id",
    "sidecar_type",
    "import_ok",
    "motion_ok",
    "export_ok",
    "roundtrip_ok",
    "overall_pass",
    "error_category",
    "error_notes",
    "export_preset",
    "output_duration_s",
    "audio_ok",
    "stabilization_ok",
]


@dataclass
class Summary:
    total_rows: int
    completed_rows: int
    pass_rows: int
    blocker_failures: int

    @property
    def pass_rate(self) -> float:
        if self.completed_rows == 0:
            return 0.0
        return self.pass_rows / self.completed_rows


def parse_csv_bool(value: str) -> bool | None:
    val = (value or "").strip().lower()
    if val in {"1", "true", "yes", "y", "pass"}:
        return True
    if val in {"0", "false", "no", "n", "fail"}:
        return False
    return None


def generate_matrix(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combos = list(
        itertools.product(
            args.camera_models,
            args.studio_versions,
            args.clips,
            args.sidecar_types,
        )
    )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()

        for camera_model, studio_version, clip_id, sidecar_type in combos:
            writer.writerow(
                {
                    "date": args.date,
                    "tester": args.tester,
                    "camera_model": camera_model,
                    "studio_version": studio_version,
                    "clip_id": clip_id,
                    "sidecar_type": sidecar_type,
                    "import_ok": "",
                    "motion_ok": "",
                    "export_ok": "",
                    "roundtrip_ok": "",
                    "overall_pass": "",
                    "error_category": "",
                    "error_notes": "",
                    "export_preset": args.export_preset,
                    "output_duration_s": "",
                    "audio_ok": "",
                    "stabilization_ok": "",
                }
            )

    print(f"Wrote {len(combos)} matrix rows to {output_path}")


def summarize_matrix(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    completed_rows = 0
    pass_rows = 0
    blocker_failures = 0

    for row in rows:
        overall = parse_csv_bool(row.get("overall_pass", ""))
        if overall is None:
            continue

        completed_rows += 1
        if overall:
            pass_rows += 1
        else:
            blocker_failures += 1

    summary = Summary(
        total_rows=len(rows),
        completed_rows=completed_rows,
        pass_rows=pass_rows,
        blocker_failures=blocker_failures,
    )

    print(f"Input file: {input_path}")
    print(f"Total rows: {summary.total_rows}")
    print(f"Completed rows: {summary.completed_rows}")
    print(f"Pass rows: {summary.pass_rows}")
    print(f"Blocker failures: {summary.blocker_failures}")
    print(f"Pass rate: {summary.pass_rate * 100:.2f}%")

    if summary.completed_rows == 0:
        print("Decision: INSUFFICIENT DATA (no completed rows)")
        return

    sidecar_go = summary.pass_rate >= args.pass_rate_threshold and blocker_failures == 0
    if sidecar_go:
        print("Decision: GO (sidecar-first remains primary)")
    else:
        print("Decision: NO-GO (trigger SDK-backed renderer pivot investigation)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate compatibility matrix rows")
    gen.add_argument("--output", default="artifacts/validation/compatibility-matrix.csv")
    gen.add_argument("--date", default=str(date.today()))
    gen.add_argument("--tester", default="")
    gen.add_argument("--camera-models", nargs="+", required=True)
    gen.add_argument("--studio-versions", nargs="+", required=True)
    gen.add_argument("--clips", nargs="+", required=True)
    gen.add_argument("--sidecar-types", nargs="+", default=DEFAULT_SIDECAR_TYPES)
    gen.add_argument("--export-preset", default="")
    gen.set_defaults(func=generate_matrix)

    summ = subparsers.add_parser("summarize", help="Summarize matrix and print go/no-go")
    summ.add_argument("--input", default="artifacts/validation/compatibility-matrix.csv")
    summ.add_argument("--pass-rate-threshold", type=float, default=0.95)
    summ.set_defaults(func=summarize_matrix)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
