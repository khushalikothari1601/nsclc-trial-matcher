# Eval harness

Measures the quality of the one probabilistic step in the matcher: the LLM parse from
a trial's free-text eligibility criteria into a structured `TrialCriteria`. Everything
downstream of that (patient x criteria matching) is deterministic and is covered by
`../tests/` with pytest, not here.

## Why this is scoped to the parse

The system has three layers, each with the right instrument:

| Layer | What it does | How it can fail | Tested by |
|-------|--------------|-----------------|-----------|
| `parse_criteria` | raw text -> `TrialCriteria` | LLM misreads / drops / invents criteria | **this harness** |
| `matcher` | patient x `TrialCriteria` -> `MatchResult` | logic bugs | pytest (`../tests/`) |
| `pipeline` / `app` | glue, fetch, cache | integration bugs | smoke tests |

## Status

Complete. 20 trials are labeled in `gold/`. Two runs of 3 parses per trial are saved:
`results_v0.json` (first run) and `results.json` (after the fixes it led to, rescored
against the final labels). The headline numbers and an analysis of the errors are in the
[main README](../README.md#results).

## How scoring works

`scorer.py` compares a predicted `TrialCriteria` to a hand-labeled gold one, field by
field, using a different strategy per field *kind* because they fail differently:

- **scalars** — exact match, with `missed` / `hallucinated` / `wrong_value` tracked
  separately (a dropped criterion and an invented one are different failures).
- **sets** — precision/recall after normalization, **reusing `matcher.py`'s own
  `_norm_stage` / `_norm_histology`** so a synonym the matcher handles isn't scored as
  an error. Terms that don't map to a known stage or histology ("mixed tumors with small
  cell elements") align by word overlap, as do the prior-treatment lists, which hold
  free-text clauses.
- **keyed object lists** (biomarkers, labs) — aligned by gene / lab_name, then
  sub-fields compared. A flipped biomarker requirement direction is flagged
  **critical** because that single field flips eligibility. A biomarker's free-text
  `detail` is not scored.
- **unstructured buckets** — scored in both directions: over-structuring (model
  confidently structured text that should've been flagged for review — dangerous) vs
  over-flagging (model punted structurable text to review — makes the tool useless).
  Items are near-verbatim by design, so they align by word overlap, not string equality.
  A whole list rarely matches exactly, so the report also gives the share of items
  matched in each direction.

Every field score carries a **severity** so `report.py` can rank failures by how much
they'd distort a real eligibility decision, not just count them.

Most fields are empty on any given trial, and null-vs-null counts as correct, so
accuracy over all fields looks excellent even when the parse is poor. `report.py` leads
with accuracy over **populated** fields (gold or prediction has a value) for that reason.

## Running it

From the repo root, with `ANTHROPIC_API_KEY` set:

```bash
# 1. Label ~15-25 trials into eval/gold/  (see WORKFLOW.md and LABELING.md)
python eval/label.py NCT01234567   # fetch and pre-fill, then label by hand
python eval/validate.py            # catch broken files and likely mistakes

# 2. Parse each trial N times and score every run
python eval/run_eval.py --runs 3

# 3. Summarize
python eval/report.py
```

Files with no medical fields filled in are skipped. Results are saved after every trial,
so an interrupted run keeps what it finished. `--model` evaluates a different model.

Each run also saves the model's raw parse, so after a change to the scorer or the gold
labels you can rescore without calling the model again:

```bash
python eval/rescore.py      # rescores eval/results.json in place
```

`--runs 3` (the default) parses each trial three times, because the parse is
nondeterministic and a field that's "correct 1 of 3 runs" is a coin flip you want to
see, not average away. `report.py` surfaces those flaky fields explicitly.

## What the report shows

`report.py` prints accuracy over populated fields, reliability by field type, top failure
modes (critical first), and fields that changed between runs. With fewer than 15 trials
it labels the numbers a smoke test.

## Limitations

- The gold set is single-annotator with no adjudication (see LABELING.md), so it isn't a
  validated gold standard. Labeling a few trials twice on different days and reporting
  self-agreement would be a cheap way to strengthen it.
- Multi-cohort trials don't fit the schema: cohort-specific criteria go to the
  unstructured buckets.
- Word-overlap alignment is approximate. It was tuned so distinct criteria within the
  labeled trials don't align with each other, which is a small sample.
