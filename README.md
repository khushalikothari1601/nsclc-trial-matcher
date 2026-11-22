# NSCLC Trial Matcher (prototype)

Enter a structured NSCLC (non-small cell lung cancer) patient profile and see which
currently-recruiting trials on ClinicalTrials.gov the patient likely matches, based on
an LLM-parsed structured version of each trial's eligibility criteria.

> **This is a personal prototype / decision-support tool, not a clinical or
> HIPAA-compliant system.** Do not enter real patient information (PHI) -- use only
> synthetic or de-identified test profiles. Output is not clinical advice and has not
> been validated for clinical use.

## Setup

```bash
cd nsclc-trial-matcher
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
export $(grep -v '^#' .env | xargs)   # or just `export ANTHROPIC_API_KEY=...`
```

## Usage

1. Fetch and parse a batch of trials (does the ClinicalTrials.gov fetch + one LLM
   parse per new trial, caching results in `data/cache.sqlite3`):

   ```bash
   python pipeline.py --max-trials 50 --status RECRUITING
   ```

   Re-running is cheap -- trials already in the cache (by NCT ID + eligibility-text
   hash) are not re-parsed.

2. Launch the app:

   ```bash
   streamlit run app.py
   ```

   Fill in the patient form and submit to see matches against whatever is in the
   local cache. Use the sidebar "Refresh trial data" button for small incremental
   pulls, or re-run `pipeline.py` from the CLI for a larger batch.

## How matching works

Each trial's free-text eligibility criteria, plus the age and sex limits that
ClinicalTrials.gov stores separately, is parsed once by Claude into a fixed schema (age, ECOG, histology, stage, biomarkers, PD-L1, prior therapy, brain mets,
labs, etc.), with anything that doesn't map cleanly into that schema kept as
near-verbatim text for manual review rather than being dropped or guessed at.

Matching itself is deterministic, plain-Python field comparison (see `matcher.py`) --
not another LLM call -- so results are explainable and reproducible. A trial is:

- **Not Eligible** if any structured criterion is explicitly violated.
- **Possibly Eligible - Needs Review** if any criterion couldn't be evaluated (missing
  patient data) or the trial has unstructured eligibility text that wasn't captured
  by the schema.
- **Likely Eligible** only if every structured criterion passed and there's no
  unstructured text left to review.

Missing patient data never counts as a failure -- only an explicit contradiction does.
The same goes for trial text the matcher can't interpret. Terms like "locally advanced",
ULN-relative lab limits, drug-class wording ("any KRAS G12C inhibitor") and unnamed
variants ("EGFR positive" against "exon 20 insertion") all evaluate to unknown, so those
trials go to review rather than being ruled out or waved through.

## Testing and evaluation

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The tests need no API key. They cover the matcher, the eval scorer and runner (with the
model stubbed out), and the trial-text formatting.

The one probabilistic step, the LLM parse, is graded by the harness in [`eval/`](eval/),
which scores it field by field against hand-labeled trials, weighted by how much each
error would distort an eligibility decision, across repeated runs. See
[`eval/README.md`](eval/README.md).

## Results

Two runs in November 2026, each parsing all 20 labeled trials 3 times with `claude-opus-5`
(the pipeline default). All 120 parses completed. Raw scores are in
[`eval/results_v0.json`](eval/results_v0.json) and [`eval/results.json`](eval/results.json),
and `python eval/report.py` reprints either summary.

| | v0 | v1 |
|---|---|---|
| Accuracy on populated fields | 57.5% | **68.5%** |
| Accuracy on all fields (inflated by fields neither side fills) | 72.6% | 82.2% |
| Critical errors (ones that could flip an eligibility decision), per 60 parses | 122 | **40** |
| Scalar fields (ECOG, age, PD-L1, flags) | 77% | 91% |
| Biomarker and lab entries | 66% | 77% |
| List fields (stage, histology, prior treatments) | 42% | 60% |
| Unstructured items matched, gold → parse / parse → gold | 78% / 88% | 87% / 89% |
| Unstructured lists, scored as whole lists | 19% | 25% |

### What changed between v0 and v1

After v0, every critical error was reviewed (47 distinct trial-and-field patterns). The
review led to three kinds of change:

- **Parse prompt:** don't force a gene or variant the schema can't express (KRAS G12D,
  NRAS) onto the nearest value. Keep cohort-specific rules out of trial-wide fields. Read
  "active CNS metastases excluded" as `treated_stable`. Give each prior-treatment rule
  one canonical place.
- **Labels:** the same rules were written into [`LABELING.md`](eval/LABELING.md) and
  applied to all 20 files, deciding each case from the trial text. Repeats of structured
  fields in the unstructured lists were removed. Two labels that broke the guide were
  corrected: "ALK/ROS1-positive" had been labeled as requiring both, and a prior-radiation
  exclusion sat in the unstructured list.
- **Scorer:** free-text stage and histology items now align by wording, so "containing"
  and "with" no longer count as a mismatch. "M1 disease" counts as stage IV. Unstructured
  lists are also reported item by item.

v1 saved the model's raw output, so the same v1 parses can be regraded under each set of
labels and scorer code. That separates the gain:

| v1 parses graded with | Accuracy, populated fields | Critical errors |
|---|---|---|
| old labels, old scorer | 61.8% | 89 |
| new labels, old scorer | 67.7% | 44 |
| new labels, new scorer | 68.5% | 40 |

Against v0's 57.5% and 122, that's about **+4 points and −33 critical errors from the
prompt**, **+6 points and −45 from label consistency**, and **+1 point and −4 from the
scorer**. These are single runs, so allow a point or two of run-to-run noise in each step.

The targeted prompt fixes worked:
- KRAS G12D was forced to G12C in 2 of 3 v0 runs and 0 of 3 in v1.
- The basket trial's nine trial-wide biomarker entries are gone.
- NCT07286149's brain-metastasis rule is now read correctly.

### What's left

The 40 remaining critical errors come from 22 trial-and-field patterns. The parser still:
- reads NCT06170788's "known active CNS metastases" exclusion as "no brain mets allowed"
  in 2 of 3 runs;
- lists drug classes that "treatment-naive" already implies (NCT05011487, NCT06864624),
  and sometimes drops `treatment_naive_required: true`;
- flips NCT04973293's EGFR rule to `required_negative` in 1 of 3 runs.

The rest are mostly free-text exclusions worded or split differently, and a few places
where the labeling rules leave room for judgment (for example, where "N3 disease" goes).

**Interpreting the numbers:** accuracy here means agreement with one annotator's labels,
following a written guide, not clinical accuracy. Changing the labels and the scorer
between runs is part of why v1 is higher. The regrading table shows how much each part
contributed, and every label change was decided from the trial text and the written rules.

**How the labels were made:** all 20 trials were labeled by Khushali Kothari. One
(NCT06092086) was drafted by Claude at her request and reviewed by her. Claude also
converted several files' formatting to the schema, and applied the consistency rules after
she approved each change list. There is no second annotator.

**Next steps:**
1. Have a second person label a subset, to measure agreement.
2. Add more trials, especially multi-cohort and basket trials, and study the
   lowest-scoring ones (NCT05613452 at 35% and NCT06172478 at 37%).
3. Extend the schema for genes and variants it can't represent yet (KRAS G12D, NRAS) and
   for "either/or" biomarker requirements.

## Known limitations

- Parse accuracy is measured on 20 trials with one annotator (see Results).
- Most real matches land in "Needs Review": real trials nearly always have free text the
  schema can't hold.
- Multi-cohort trials are flattened to one set of criteria.
- Prior-treatment matching is string overlap with no drug-class knowledge, and doesn't
  evaluate time windows ("within 4 weeks").
- Patient biomarker variant isn't collected by the form yet, so variant-specific
  requirements always go to review.
