"""Summarize results.json.

Three views, each answering a different question about the parse:

  1. RELIABILITY BY FIELD TYPE   -- "where is the parse trustworthy vs shaky?"
       Accuracy grouped by scalar / set / object_list / unstructured, because those
       fail for different reasons and a single blended number hides that.

  2. TOP FAILURE MODES           -- "what does it get wrong most, and how badly?"
       Per-field error counts, ranked, with critical (eligibility-flipping) errors
       isolated at the top.

  3. CROSS-RUN CONSISTENCY        -- "is it stable, or right-by-luck?"
       For each (trial, field), how often the N runs agreed with gold. A field that's
       correct 3/3 is solid; 1/3 is a coin flip you shouldn't trust.
"""

from __future__ import annotations

import argparse
import json
import re
import os
from collections import defaultdict

MIN_MEANINGFUL_CASES = 15


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def unstructured_items(runs: list[dict]) -> tuple[float, float] | None:
    """Share of gold unstructured items matched by the parse, and of parsed items matched by
    gold. Uses n_missing/n_extra when present, else the counts in the note (older files)."""
    g = m = p = x = 0
    for r in runs:
        for f in r.get("field_scores", []):
            if f["field_type"] != "unstructured":
                continue
            missing, extra = f.get("n_missing"), f.get("n_extra")
            if missing is None:
                note = f.get("note") or ""
                mm = re.search(r"(\d+) review item\(s\) not flagged", note)
                xx = re.search(r"(\d+) extra", note)
                missing, extra = int(mm.group(1)) if mm else 0, int(xx.group(1)) if xx else 0
            g += f["gold"]
            m += f["gold"] - missing
            p += f["predicted"]
            x += f["predicted"] - extra
    if not g or not p:
        return None
    return m / g, x / p


def reliability_by_type(runs: list[dict]) -> dict[str, tuple[int, int]]:
    """Accuracy per field type over populated fields; null-vs-null agreement says nothing."""
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # type -> [correct, total]
    for r in runs:
        for f in r.get("field_scores", []):
            if not f.get("populated", True):
                continue
            t = f["field_type"]
            tally[t][1] += 1
            if f["outcome"] == "correct":
                tally[t][0] += 1
    return {t: (c, n) for t, (c, n) in tally.items()}


def failure_modes(runs: list[dict]) -> list[tuple[str, str, int]]:
    # (field_name, severity) -> count of non-correct outcomes
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in runs:
        for f in r.get("field_scores", []):
            if f["outcome"] != "correct":
                counts[(f["field_name"], f["severity"])] += 1
    ranked = sorted(counts.items(), key=lambda kv: (kv[0][1] != "critical", -kv[1]))
    return [(name, sev, n) for (name, sev), n in ranked]


def consistency(runs: list[dict]) -> list[tuple[str, str, int, int]]:
    # (nct_id, field_name) -> [correct_runs, total_runs]
    agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in runs:
        nct = r["nct_id"]
        for f in r.get("field_scores", []):
            key = (nct, f["field_name"])
            agg[key][1] += 1
            if f["outcome"] == "correct":
                agg[key][0] += 1
    flaky = [(nct, fn, c, t) for (nct, fn), (c, t) in agg.items() if t > 1 and 0 < c < t]
    return sorted(flaky, key=lambda x: x[2] / x[3])


def bar(frac: float, width: int = 24) -> str:
    filled = round(frac * width)
    return "#" * filled + "." * (width - filled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "results.json"))
    args = ap.parse_args()
    data = load(args.results)
    runs = data["runs"]
    n_cases = data.get("n_cases", "?")
    n_runs = data.get("n_runs", "?")

    scored = [r for r in runs if "field_scores" in r]
    errors = [r for r in runs if "parse_error" in r]
    accs = [r["accuracy"] for r in scored]
    overall = sum(accs) / len(accs) if accs else 0.0
    populated = [r["populated_accuracy"] for r in scored if "populated_accuracy" in r]
    overall_populated = sum(populated) / len(populated) if populated else 0.0
    n_critical = sum(
        1 for r in scored for f in r["field_scores"] if f["outcome"] != "correct" and f["severity"] == "critical"
    )

    print("=" * 60)
    print(f"PARSE EVAL  --  {n_cases} trial(s) x {n_runs} run(s)")
    print("=" * 60)
    if data.get("model"):
        print(f"Model: {data['model']}")
    print(f"Accuracy, populated fields: {overall_populated:.1%}")
    print(f"Accuracy, all fields:       {overall:.1%}   (inflated by fields neither side mentions)")
    print(f"Critical errors:            {n_critical}")
    items = unstructured_items(scored)
    if items:
        print(f"Unstructured items matched: {items[0]:.0%} of gold, {items[1]:.0%} of parse")
    print(f"Runs:                       {len(scored)} scored, {len(errors)} parse errors\n")
    if isinstance(n_cases, int) and n_cases < MIN_MEANINGFUL_CASES:
        print(f"NOTE: only {n_cases} trial(s). Treat these numbers as a smoke test, not a measurement.\n")

    print("Reliability by field type (populated fields)")
    print("-" * 60)
    for t, (c, n) in sorted(reliability_by_type(scored).items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1)):
        frac = c / n if n else 0
        print(f"  {t:13} {bar(frac)} {frac:5.0%}  ({c}/{n})")

    print("\nTop failure modes (critical first)")
    print("-" * 60)
    fm = failure_modes(scored)
    if not fm:
        print("  none -- every field correct on every run.")
    for name, sev, n in fm[:12]:
        tag = " <-- eligibility-flipping" if sev == "critical" else ""
        print(f"  [{sev:8}] {name:30} {n:2} miss(es){tag}")

    print("\nFlaky fields (right on some runs, wrong on others)")
    print("-" * 60)
    fl = consistency(scored)
    if not fl:
        print("  none -- fields were stable across runs (good, or too few runs).")
    for nct, fn, c, t in fl[:12]:
        print(f"  {nct:14} {fn:28} correct {c}/{t} runs")

    print("\n" + "=" * 60)
    print("Summary:")
    types = reliability_by_type(scored)
    if types:
        best = max(types.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))
        worst = min(types.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))
        crit = sum(1 for _, s, _ in fm if s == "critical")
        print(f'  "Parse was strongest on {best[0]} fields and weakest on {worst[0]};'
              f' {crit} field(s) showed eligibility-flipping errors worth fixing first."')
    print("=" * 60)


if __name__ == "__main__":
    main()
