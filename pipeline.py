"""Orchestrates fetch -> cache-check -> LLM-parse -> cache-save. Runnable standalone
from the CLI for a full batch, or called from app.py for a small incremental refresh."""

import argparse
import sys

from cache import cache_stats, get_cached_criteria, hash_text, save_criteria
from fetch_trials import DEFAULT_CONDITION, fetch_nsclc_trials
from parse_criteria import DEFAULT_MODEL, parse_trial_criteria


def fetch_and_parse(
    max_trials: int = 50,
    condition: str = DEFAULT_CONDITION,
    status: str = "RECRUITING",
    model: str = DEFAULT_MODEL,
    on_progress=None,
) -> dict:
    """Fetch up to max_trials trials and ensure each has parsed, cached criteria.

    Returns a summary dict: {"fetched": N, "cache_hits": N, "parsed": N, "errors": [...]}.
    on_progress, if given, is called with (index, total, nct_id, action) for UI updates.
    """

    trials = fetch_nsclc_trials(max_trials=max_trials, condition=condition, status=status)

    cache_hits = 0
    parsed = 0
    errors: list[str] = []

    for i, trial in enumerate(trials, start=1):
        text_hash = hash_text(trial.eligibility_criteria)
        cached = get_cached_criteria(trial.nct_id, text_hash)

        if cached is not None:
            cache_hits += 1
            if on_progress:
                on_progress(i, len(trials), trial.nct_id, "cache_hit")
            continue

        if on_progress:
            on_progress(i, len(trials), trial.nct_id, "parsing")

        try:
            criteria = parse_trial_criteria(
                nct_id=trial.nct_id,
                trial_title=trial.brief_title,
                overall_status=trial.overall_status,
                eligibility_text=trial.eligibility_criteria,
                model=model,
            )
            save_criteria(trial.nct_id, text_hash, criteria, model)
            parsed += 1
        except Exception as exc:  # noqa: BLE001 - surface any parse/API failure per-trial
            errors.append(f"{trial.nct_id}: {exc}")

    return {"fetched": len(trials), "cache_hits": cache_hits, "parsed": parsed, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Fetch and parse NSCLC trials from ClinicalTrials.gov")
    parser.add_argument("--max-trials", type=int, default=50)
    parser.add_argument("--condition", default=DEFAULT_CONDITION)
    parser.add_argument("--status", default="RECRUITING")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    def report(i, total, nct_id, action):
        print(f"[{i}/{total}] {nct_id}: {action}", file=sys.stderr)

    summary = fetch_and_parse(
        max_trials=args.max_trials,
        condition=args.condition,
        status=args.status,
        model=args.model,
        on_progress=report,
    )

    print(
        f"\nFetched {summary['fetched']} trials -- "
        f"{summary['cache_hits']} cache hits, {summary['parsed']} newly parsed, "
        f"{len(summary['errors'])} errors."
    )
    for err in summary["errors"]:
        print(f"  ERROR: {err}", file=sys.stderr)

    print(f"\nCache now holds {cache_stats()['trial_count']} trials.")


if __name__ == "__main__":
    main()
