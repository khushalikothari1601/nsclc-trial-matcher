# Fast labeling workflow

Per trial, three steps. Only step 2 needs your brain.

## 1. Fetch + pre-fill (mechanical, automatic)

```bash
python eval/label.py NCT01234567
```

Fetches the trial from the ClinicalTrials.gov API and writes
`eval/gold/NCT01234567.json` with the raw eligibility text (exactly what the
pipeline sends the parser) and all metadata — title, status, age, sex — already
filled. Everything mechanical is done.

## 2. Label the medical fields (your judgment)

Open the file. Fill in ONLY the structured medical fields by reading the raw text:
ecog_max, stage, histology, biomarkers, pdl1, labs, treatment history, brain mets,
and the two unstructured buckets. Log every judgment call in `_notes`.

Follow LABELING.md for the rules. The calls that recur:
- structure only what's true for ALL patients; cohort-specific criteria -> unstructured
- biomarker direction (required_positive vs excluded_if_positive) is the highest-stakes field
- anything that doesn't map cleanly to a field -> unstructured_*, near-verbatim
- ULN-relative labs (e.g. "<= 1.5 x ULN"): label them faithfully. The matcher reports
  them as UNKNOWN and the validator notes it

## 3. Validate (mechanical, automatic)

```bash
python eval/validate.py NCT01234567      # one file
python eval/validate.py                  # all files
```

ERRORS must be fixed (broken JSON / schema). WARNINGS are likely-mistake patterns —
read each and decide. A clean file prints `OK`.

## When all ~20 are done

```bash
python eval/validate.py          # everything OK / warnings resolved
python eval/run_eval.py --runs 3
python eval/report.py
```

## Tips for a good set

- Aim for variety, not ease: a few simple trials, several messy ones (prose biomarker
  rules, nested logic, dense labs, multi-cohort). The failures are the point.
- Keep judgment calls CONSISTENT across trials. If "Stage IIIB or IV" is two entries
  in trial 3, it's two entries in trial 15.
- Keep a running list of schema limitations you hit (multi-cohort, ULN labs, etc.).
