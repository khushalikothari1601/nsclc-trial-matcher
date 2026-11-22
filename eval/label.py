#!/usr/bin/env python3
"""Fast gold-labeling helper.

Usage:
    python eval/label.py NCT04973293

Fetches the trial from the ClinicalTrials.gov API v2 and writes a pre-filled
gold file to eval/gold/<NCT>.json with everything mechanical already done:
  - raw eligibility text, built by the same function the pipeline uses, so
    _raw_eligibility_text is exactly what the parser sees in production,
  - nct_id, trial_title, overall_status, min_age_years, max_age_years, sex
    filled from the structured API fields,
  - every medical field left blank for YOU to fill in.

You then open the file and label only the structured medical fields (ecog, stage,
histology, biomarkers, labs, unstructured buckets). Run eval/validate.py when done.

Note: the API sometimes gives age as "18 Years" / "N/A". This script parses the
number out; if it can't, it leaves the field null and notes it so you can fill it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fetch_trials import format_eligibility_text  # noqa: E402

API = "https://clinicaltrials.gov/api/v2/studies/{}"
GOLD_DIR = os.path.join(os.path.dirname(__file__), "gold")


def fetch(nct_id: str) -> dict:
    url = API.format(nct_id)
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "gold-labeler/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def parse_age(s: str | None) -> float | None:
    # "18 Years" -> 18 ; "6 Months" -> 0.5 ; "N/A"/None -> None
    if not s or s.strip().upper() in {"N/A", "NONE", ""}:
        return None
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(year|month|week|day)", s, re.I)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2).lower()
    factor = {"year": 1, "month": 1 / 12, "week": 1 / 52, "day": 1 / 365}[unit]
    return round(val * factor, 3)


def sex_to_enum(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip().upper()
    return {"ALL": "all", "MALE": "male", "FEMALE": "female"}.get(s)


def build(nct_id: str, data: dict) -> dict:
    ps = data.get("protocolSection", {})
    elig = ps.get("eligibilityModule", {})
    ident = ps.get("identificationModule", {})
    status = ps.get("statusModule", {})

    if not elig.get("eligibilityCriteria", "").strip():
        print(f"  ! WARNING: no eligibilityCriteria text found for {nct_id}", file=sys.stderr)

    min_age = parse_age(elig.get("minimumAge"))
    max_age = parse_age(elig.get("maximumAge"))
    sex = sex_to_enum(elig.get("sex"))
    raw = format_eligibility_text(elig)

    notes = []
    if min_age is None and elig.get("minimumAge"):
        notes.append(f"AGE PARSE FAILED for minimumAge={elig.get('minimumAge')!r} - fill min_age_years by hand.")
    if max_age is None and elig.get("maximumAge"):
        notes.append(f"AGE PARSE FAILED for maximumAge={elig.get('maximumAge')!r} - fill max_age_years by hand.")

    return {
        "_raw_eligibility_text": raw,
        "_source": f"https://clinicaltrials.gov/study/{nct_id}",
        "_labeler": "Khushali Kothari",
        "_notes": " ".join(notes) if notes else "TODO: log your judgment calls here as you label.",
        "nct_id": nct_id,
        "trial_title": ident.get("briefTitle", ""),
        "overall_status": status.get("overallStatus", ""),
        "min_age_years": min_age,
        "max_age_years": max_age,
        "sex": sex,
        "ecog_max": None,
        "histology_included": [],
        "histology_excluded": [],
        "stage_included": [],
        "stage_excluded": [],
        "biomarkers": [],
        "pdl1_tps_min": None,
        "pdl1_tps_max": None,
        "prior_therapy_lines_min": None,
        "prior_therapy_lines_max": None,
        "treatment_naive_required": None,
        "required_prior_treatments": [],
        "excluded_prior_treatments": [],
        "brain_mets_status": None,
        "measurable_disease_required": None,
        "organ_function_labs": [],
        "unstructured_inclusion": [],
        "unstructured_exclusion": [],
        "parse_confidence": None,
        "manual_review_flag": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nct_id", help="e.g. NCT04973293")
    ap.add_argument("--force", action="store_true", help="overwrite an existing gold file")
    args = ap.parse_args()

    nct = args.nct_id.strip().upper()
    if not re.fullmatch(r"NCT\d{8}", nct):
        sys.exit(f"'{nct}' doesn't look like an NCT id (expected NCT + 8 digits).")

    os.makedirs(GOLD_DIR, exist_ok=True)
    out = os.path.join(GOLD_DIR, f"{nct}.json")
    if os.path.exists(out) and not args.force:
        sys.exit(f"{out} already exists. Use --force to overwrite.")

    try:
        data = fetch(nct)
    except Exception as exc:
        sys.exit(f"Fetch failed for {nct}: {exc}")

    gold = build(nct, data)
    with open(out, "w") as fh:
        json.dump(gold, fh, indent=2)

    print(f"Wrote {out}")
    print(f"  title:  {gold['trial_title'][:64]}")
    print(f"  status: {gold['overall_status']}  |  age: {gold['min_age_years']}-{gold['max_age_years']}  |  sex: {gold['sex']}")
    print(f"  raw text: {len(gold['_raw_eligibility_text'])} chars")
    print("  -> now open it and fill in the structured medical fields, then run eval/validate.py")


if __name__ == "__main__":
    main()
