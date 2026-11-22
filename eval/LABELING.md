# Gold labeling guide

The harness compares the LLM's parse against a small set of **hand-labeled** trials
(the "gold set"). The scores are only as trustworthy as this labeling is consistent,
so read this once before you start — the rules here are what stop 20 trials from
being labeled 20 slightly different ways.

## What you're producing

One JSON file per trial in `gold/`, named `NCT********.json`. Each file is a filled-in
copy of `gold_template.json`: the trial's raw eligibility text under
`_raw_eligibility_text`, plus the **correct** `TrialCriteria` you believe that text
maps to. The harness re-parses the raw text with the LLM and scores its output against
your labels.

## How to label one trial

1. Run `python eval/label.py NCT<the id>`. It writes `gold/NCT<the id>.json` with the raw
   text and metadata filled in, using the same text the pipeline sends the parser. (If
   you can't reach the API, copy `gold_template.json` and paste the criteria verbatim.)
2. Don't edit `_raw_eligibility_text` — the model must parse the same input you
   labeled from.
3. Fill the structured fields with what the text **explicitly** says. Then put every
   criterion that does NOT map cleanly onto a field into `unstructured_inclusion` /
   `unstructured_exclusion`, near-verbatim.

## Consistency rules (the important part)

These are the judgment calls that, left unwritten, quietly corrupt a gold set. Decide
them once — here — and apply them every time.

- **Only what's stated.** Never infer. If the trial doesn't mention PD-L1, leave both
  `pdl1_tps_*` as `null`. Labeling something the text doesn't say turns a correct model
  parse into a false "hallucination."
- **`null` vs `0`.** `null` = "the trial doesn't mention this." `0` = "the trial
  explicitly sets this to zero." They are different and the scorer treats them
  differently. Don't use `0` as a stand-in for "not mentioned."
- **ECOG.** `ecog_max` is the highest allowed value. "ECOG 0-1" -> `ecog_max: 1`.
  "ECOG 0-2" -> `2`.
- **Stage.** One list entry per stage token as written: "Stage IIIB or IV" ->
  `["IIIB", "IV"]`, not `["IIIB or IV"]`. Don't expand ranges the trial didn't
  enumerate.
- **Histology.** Use the trial's words ("adenocarcinoma", "squamous"). The matcher's
  normalizer handles synonyms, so don't pre-normalize — label what's written.
- **Biomarkers — direction matters most.** This is the highest-stakes field:
  - required to BE positive -> `required_positive`
  - required to BE negative / wild-type -> `required_negative`
  - a positive result EXCLUDES the patient -> `excluded_if_positive`
  "Excludes patients with known EGFR mutations" is `excluded_if_positive`, NOT
  `required_negative`. When the trial names a specific variant ("EGFR exon 19 deletion"), put it
  verbatim in `detail`.
- **Multi-cohort trials.** Structure a criterion only if it applies to every
  participant. Anything that applies to one cohort, phase or tumor type goes in the
  unstructured lists, prefixed with the cohort ("Cohort 2: …").
- **Genes and variants the schema can't express.** `KRAS_G12C` means only G12C. A
  requirement for KRAS G12D, NRAS or any other unlisted gene or variant goes in the
  unstructured lists, never under the nearest listed value.
- **Brain metastases.** `brain_mets_status` is the most permissive state the trial
  accepts. If only active, untreated, symptomatic or unstable mets are excluded, it's
  `treated_stable`. Use `none` only when all brain mets are excluded, and leave it null
  when the trial doesn't address them.
- **Prior treatment — one place per rule.** Each kind of rule has exactly one home,
  chosen by what the matcher can check:

  | The trial says… | Label it as | Don't also |
  |---|---|---|
  | "No prior systemic therapy" / "treatment-naive" (optionally "for advanced disease") | `treatment_naive_required: true` | list the classes it implies in `excluded_prior_treatments`, or set `prior_therapy_lines_*` to 0 |
  | …with a setting exception ("prior adjuvant therapy allowed if >12 months ago") | `treatment_naive_required: true`, exception in `unstructured_inclusion` | |
  | "At least N prior lines" / "progressed after 1–2 lines" | `prior_therapy_lines_min` / `_max` | set `treatment_naive_required: false` |
  | A specific prior drug or class is required ("prior platinum chemotherapy") | `required_prior_treatments`, one short entry each | set `treatment_naive_required: false` |
  | "Prior therapy required" with no number and no named treatment | `treatment_naive_required: false` | |
  | A requirement that applies only to one cohort or one condition ("Cohort 2: progressed on osimertinib") | `unstructured_inclusion` | put it in `required_prior_treatments` |
  | A specific prior drug or class is excluded, with no time window ("prior KRAS G12C inhibitor", "prior anti-PD-1/PD-L1", "prior radiotherapy") | `excluded_prior_treatments`, one short entry each | |
  | Washout or time-window rules ("any anticancer therapy within 4 weeks", "radiotherapy within 14 days", "major surgery within 28 days") | `unstructured_exclusion` | put them in `excluded_prior_treatments` |
  | Concomitant medications, vaccines, other trials ("strong CYP3A4 inhibitors", "live vaccine within 30 days", "another investigational drug within 4 weeks") | `unstructured_exclusion` | put them in `excluded_prior_treatments` |

- **Labs.** One entry per lab with operator + value + unit. "ANC >= 1.5 x10^9/L" ->
  `{"lab_name": "ANC", "operator": ">=", "value": 1.5, "unit": "x10^9/L"}`.
- **Unstructured buckets.** If a criterion can't be captured by a field WITHOUT
  losing meaning, it belongs here — don't force it. Deciding this line consistently
  is itself part of what the harness measures (over- vs under-structuring), so be
  deliberate: force-fitting a criterion and punting an easily-structured one are both
  errors.
- **`manual_review_flag`** is set automatically from the unstructured buckets by
  `from_extracted`; leave it `false` in gold files, the harness doesn't score it.

## Choosing which trials to label

Don't label 20 easy ones — you'll learn nothing. Aim for ~15–25 trials, weighted
toward the phrasings you suspect the model fumbles:

- a few clean, simple trials (baseline — the model should ace these)
- several with **prose** biomarker rules ("patients whose tumors harbor an
  activating EGFR mutation are not eligible")
- some with **nested / "or"** logic ("Stage IIIB not amenable to definitive
  chemoradiation, or Stage IV")
- some with dense **lab panels** and unusual units
- one or two genuinely messy ones with lots of unstructurable text

The failures you find are the whole point — a gold set of only easy trials produces a
flattering, useless 98%.

## Limits of this gold set

There is a **single annotator** and **no second-labeler adjudication**, so describe it as
"~20 trials, single annotator, no inter-annotator agreement," not as a validated gold
standard. To strengthen it, label a handful of trials twice on different days and report
self-agreement.

Labels should come from a person reading the trial. A model-written gold set would be
grading a model against itself.
