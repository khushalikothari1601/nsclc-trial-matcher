#!/usr/bin/env python3
"""Validate hand-labeled gold files before running the harness.

Usage:
    python eval/validate.py                 # check every file in eval/gold/
    python eval/validate.py NCT04973293     # check one

Two levels of check:
  ERRORS  -- the file is broken or won't load (invalid JSON, wrong types, bad enum
             values). Must fix; the harness would skip or crash on these.
  WARNINGS -- the file loads but shows a pattern that is *usually* a labeling mistake
             (e.g. an age labeled that never appears in the raw text, or empty
             unstructured buckets on a trial with many exclusion lines). Not always
             wrong -- read and decide.

Exit code is nonzero if any ERROR is found, so you can wire it into a pre-commit hook.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pydantic import ValidationError  # noqa: E402

from run_eval import META_KEYS, is_labeled  # noqa: E402
from schema import TrialCriteria  # noqa: E402

GOLD_DIR = os.path.join(os.path.dirname(__file__), "gold")
PLACEHOLDER_NOTES = "any judgment calls you made while labeling"


def check_file(path: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    name = os.path.basename(path)

    try:
        with open(path) as fh:
            blob = json.load(fh)
    except json.JSONDecodeError as exc:
        return ([f"{name}: INVALID JSON at line {exc.lineno} col {exc.colno}: {exc.msg}"], [])

    raw = blob.get("_raw_eligibility_text", "")
    if not raw or raw.strip().startswith("PASTE"):
        errors.append(f"{name}: _raw_eligibility_text is empty or still the placeholder.")

    # schema conformance
    criteria = {k: v for k, v in blob.items() if k not in META_KEYS}
    try:
        tc = TrialCriteria(**criteria)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        errors.append(f"{name}: schema error at '{loc}': {first['msg']}")
        return (errors, warnings)

    if not is_labeled(blob):
        warnings.append(f"{name}: no medical fields filled in -- run_eval.py will skip this file.")
        return (errors, warnings)

    # --- heuristic warnings (the mistakes actually seen while labeling) ---
    raw_low = raw.lower()

    # 1. age labeled but not present in raw text -> model can't see it
    if tc.min_age_years is not None and "age" not in raw_low and "year" not in raw_low:
        warnings.append(f"{name}: min_age_years={tc.min_age_years} but raw text mentions no age -- "
                        f"did you label from the age box without adding it to the raw text?")

    # 2. nct placeholder left in
    if "XXXX" in blob.get("nct_id", "") or "XXXX" in blob.get("_source", ""):
        warnings.append(f"{name}: nct_id / _source still contains an XXXX placeholder.")

    # 3. many exclusion lines but empty unstructured_exclusion
    excl_markers = raw_low.count(";") 
    has_excl_section = "exclusion" in raw_low
    if has_excl_section and excl_markers >= 5 and not tc.unstructured_exclusion:
        # count structured exclusion-ish fields to avoid false positives
        structured_excl = bool(tc.histology_excluded or tc.stage_excluded or tc.excluded_prior_treatments
                               or any(b.status.value == "excluded_if_positive" for b in tc.biomarkers))
        if not structured_excl:
            warnings.append(f"{name}: trial has an exclusion section with many lines but "
                            f"unstructured_exclusion is empty -- did you capture them?")

    # 4. title/status blank
    if not blob.get("trial_title"):
        warnings.append(f"{name}: trial_title is blank.")
    if not blob.get("overall_status"):
        warnings.append(f"{name}: overall_status is blank.")

    # 5. biomarker with excluded_if_positive but 'not eligible/excluded' phrasing absent (sanity)
    #    (informational only -- skipped to avoid noise)

    # 6. lab units that are ULN-relative (the matcher reports these as UNKNOWN)
    for lab in tc.organ_function_labs:
        if lab.unit and "uln" in lab.unit.lower():
            warnings.append(f"{name}: lab '{lab.lab_name}' uses a ULN-relative unit ('{lab.unit}') -- "
                            f"the matcher can't compare it to an absolute value, so it always needs review.")

    # 7. _notes left as the default TODO
    if blob.get("_notes", "").startswith("TODO") or blob.get("_notes") == PLACEHOLDER_NOTES:
        warnings.append(f"{name}: _notes still says TODO -- log your judgment calls.")

    return (errors, warnings)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nct_id", nargs="?", help="optional single NCT id to check")
    args = ap.parse_args()

    if args.nct_id:
        paths = [os.path.join(GOLD_DIR, f"{args.nct_id.strip().upper()}.json")]
    else:
        paths = sorted(glob.glob(os.path.join(GOLD_DIR, "*.json")))

    if not paths:
        sys.exit("No gold files found.")

    total_err = 0
    total_warn = 0
    for path in paths:
        if not os.path.exists(path):
            print(f"  ! {os.path.basename(path)}: file not found")
            total_err += 1
            continue
        errors, warnings = check_file(path)
        total_err += len(errors)
        total_warn += len(warnings)
        name = os.path.basename(path)
        if not errors and not warnings:
            print(f"  OK   {name}")
        for e in errors:
            print(f"  ERR  {e}")
        for w in warnings:
            print(f"  warn {w}")

    print(f"\n{len(paths)} file(s): {total_err} error(s), {total_warn} warning(s).")
    sys.exit(1 if total_err else 0)


if __name__ == "__main__":
    main()
