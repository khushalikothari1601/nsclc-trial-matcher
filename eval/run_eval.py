"""Run the LLM parse over each gold-labeled trial N times and score every run
against the gold TrialCriteria.

Why N runs per trial: the parse step is nondeterministic, so a single run hides
instability. "EGFR direction correct in 2 of 3 runs" is a finding you want to see,
not average into a single misleading number. Per-field consistency across runs is
reported by report.py.

Layout expected:
  eval/gold/NCT********.json   one hand-labeled TrialCriteria per file (see LABELING.md)

Each gold file must ALSO carry the raw eligibility text it was labeled from, so the
harness can re-parse the exact input. Store it under the reserved key "_raw_eligibility_text"
(and optional "_source" metadata); everything else in the file is the gold TrialCriteria.

Files with no medical fields filled in yet are skipped: scoring against an empty label
would count every correct extraction as invented.

Results are written after every trial, so an interrupted run keeps what it paid for.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schema import TrialCriteria  # noqa: E402
from scorer import TrialScore, score_trial  # noqa: E402

RAW_KEY = "_raw_eligibility_text"
META_KEYS = {RAW_KEY, "_source", "_notes", "_labeler", "_labeled_at"}

# Filled in mechanically by label.py, so their presence doesn't mean a human labeled the file.
PREFILLED_FIELDS = {"nct_id", "trial_title", "overall_status", "min_age_years", "max_age_years", "sex",
                    "parse_confidence", "manual_review_flag"}


def is_labeled(blob: dict) -> bool:
    """True if any medical field has been filled in."""
    return any(
        value not in (None, [], "")
        for key, value in blob.items()
        if key not in META_KEYS and key not in PREFILLED_FIELDS
    )


@dataclass
class GoldCase:
    nct_id: str
    raw_text: str
    gold: TrialCriteria
    path: str


def load_gold(gold_dir: str) -> list[GoldCase]:
    cases: list[GoldCase] = []
    for path in sorted(glob.glob(os.path.join(gold_dir, "*.json"))):
        with open(path) as fh:
            blob = json.load(fh)
        raw = blob.get(RAW_KEY)
        if not raw:
            print(f"  ! skipping {os.path.basename(path)}: no {RAW_KEY}", file=sys.stderr)
            continue
        if not is_labeled(blob):
            print(f"  ! skipping {os.path.basename(path)}: not labeled yet", file=sys.stderr)
            continue
        criteria_fields = {k: v for k, v in blob.items() if k not in META_KEYS}
        gold = TrialCriteria(**criteria_fields)
        cases.append(GoldCase(nct_id=gold.nct_id, raw_text=raw, gold=gold, path=path))
    return cases


def parse_one(nct_id: str, trial_title: str, overall_status: str, raw_eligibility_text: str,
              model: str | None = None) -> TrialCriteria:
    """Call the same parse path the pipeline uses, so the eval measures production behavior."""
    from parse_criteria import DEFAULT_MODEL, parse_trial_criteria

    return parse_trial_criteria(nct_id, trial_title, overall_status, raw_eligibility_text, model or DEFAULT_MODEL)


def _save(out_path: str, payload: dict) -> None:
    tmp = f"{out_path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, out_path)


def score_record(case: GoldCase, pred: TrialCriteria, run_idx: int) -> dict:
    """Score one parse and keep the raw prediction, so it can be rescored later without
    calling the model again (see rescore.py)."""
    ts: TrialScore = score_trial(case.gold, pred)
    return {
        "nct_id": case.nct_id,
        "run": run_idx,
        "accuracy": ts.accuracy,
        "populated_accuracy": ts.populated_accuracy,
        "n_critical": len(ts.critical_errors()),
        "field_scores": [
            {
                "field_name": f.field_name,
                "field_type": f.field_type.value,
                "outcome": f.outcome.value,
                "severity": f.severity.value,
                "gold": f.gold,
                "predicted": f.predicted,
                "note": f.note,
                "populated": f.populated,
                "n_missing": f.n_missing,
                "n_extra": f.n_extra,
            }
            for f in ts.field_scores
        ],
        "prediction": pred.model_dump(mode="json"),
    }


def run(gold_dir: str, n_runs: int, out_path: str, model: str | None = None, parse=parse_one) -> None:
    cases = load_gold(gold_dir)
    if not cases:
        print(f"No usable gold cases in {gold_dir}. Label some first (see LABELING.md).")
        return

    print(f"Loaded {len(cases)} gold case(s); {n_runs} run(s) each.\n")
    all_runs: list[dict] = []
    payload = {
        "n_runs": n_runs,
        "n_cases": len(cases),
        "model": model,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": all_runs,
    }

    for case in cases:
        for run_idx in range(n_runs):
            try:
                pred = parse(case.nct_id, case.gold.trial_title, case.gold.overall_status, case.raw_text, model)
            except Exception as exc:  # a failed parse is itself a data point
                print(f"  {case.nct_id} run {run_idx}: parse ERROR -> {exc}")
                all_runs.append({"nct_id": case.nct_id, "run": run_idx, "parse_error": str(exc)})
                continue

            record = score_record(case, pred, run_idx)
            print(f"  {case.nct_id} run {run_idx}: {record['populated_accuracy']:.0%} of populated fields, "
                  f"{record['n_critical']} critical")
            all_runs.append(record)

        _save(out_path, payload)

    print(f"\nWrote raw results to {out_path}. Run report.py to summarize.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run and score the LLM parse against gold labels.")
    ap.add_argument("--gold-dir", default=os.path.join(os.path.dirname(__file__), "gold"))
    ap.add_argument("--runs", type=int, default=3, help="parses per trial (default 3, for stability)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    ap.add_argument("--model", help="model to evaluate (default: same as the pipeline)")
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (see the Setup section of the main README).")
    from parse_criteria import DEFAULT_MODEL

    run(args.gold_dir, args.runs, args.out, args.model or DEFAULT_MODEL)


if __name__ == "__main__":
    main()
