# Labeling queue

**Done, November 2026: all 20 trials are labeled.** This file records how the set was
chosen and what makes each trial hard.

The harness needs 15–25 hand-labeled trials before its numbers mean anything. Two were
done first, and these 18 brought the set to 20.

Every file below is already in `eval/gold/` with its raw text and metadata (title,
status, age, sex) filled in, so you only do step 2 of [WORKFLOW.md](WORKFLOW.md): fill
the medical fields, log your judgment calls in `_notes`, and run the validator.

```bash
python eval/validate.py NCT07633626   # after each file; aim for "OK"
```

Trials are ordered easiest first, so the consistency rules in
[LABELING.md](LABELING.md) are settled before the hard ones. The trials were picked from
the app's cached recruiting set, for variety in phrasing. **Don't look at the app's
parse of a trial while you label it.** The point is to grade that parse, so seeing it
first would bias the label.

The time estimates scale with text length and are rough.

## Warm-up — short, mostly clean (~15 min each)

- [x] **NCT07633626** — vortioxetine in ALK-positive NSCLC on lorlatinib. Short. A
      supportive-care trial: decide what "currently receiving lorlatinib" maps to.
- [x] **NCT07104630** — pulmonary rehab in advanced lung cancer survivors. Non-drug
      baseline: expect most criteria to be unstructured, and leave fields null that
      the text doesn't address.
- [x] **NCT05011487** — neoadjuvant osimertinib + chemo, EGFR-mutant stage III. Specific
      EGFR variants (put them in `detail`) and TNM-style stage wording.
- [x] **NCT03769129** — pembrolizumab + microwave ablation. ULN-relative labs and ≥/≤
      symbols. Prior-therapy requirement worded in prose.
- [x] **NCT07227025** — amivantamab + olomorasib, KRAS G12C. Long list of other driver
      alterations: check the direction of each (`required_negative` vs
      `excluded_if_positive`). Mentions cohorts.
- [x] **NCT07286149** — MK-1084 umbrella substudy, KRAS G12C (already in progress). The
      text says "include but are not limited to", so label only what's written.
- [x] **NCT07202611** — DC/NK cell therapy. Brain-mets wording, ULN labs, and a
      prior-lines requirement in prose. Minimum age is 20, not 18.
- [x] **NCT05613452** — carbon-ion therapy, peripheral early-stage NSCLC. TNM stage
      notation and a dense lab panel.

## Medium (~25–35 min each)

- [x] **NCT06120283** — BGB-43395 (CDK4 inhibitor) alone or in combination. Short, but
      the phase 1a/1b cohorts are mostly breast cancer, with NSCLC as one of several
      tumor types. Structure only what applies to everyone.
- [x] **NCT06170788** — sacituzumab tirumotecan + pembrolizumab, first line. PD-L1
      threshold, biomarker direction for several genes, treatment-naive wording.
- [x] **NCT06908070** — chemoradiotherapy + surgery + durvalumab. Nested stage logic (N
      category qualifiers) and a long exclusion list.
- [x] **NCT07632365** — induction chemoimmunotherapy + chemoradiotherapy + consolidation,
      unresectable stage III. Treatment-naive wording, drivers listed as exclusions
      ("co-occurrence of EGFR mutation, ALK or ROS-1 rearrangement"), and a lab panel in
      one sentence with g/L units and an anticoagulation exception. Replaces NCT05468242,
      whose status is UNKNOWN.
- [x] **NCT06864624** — perioperative sunvozertinib, stage II–IIIB, EGFR exon 20
      insertion. A stage list with a qualifier ("IIIB (N2)"), a variant-specific
      requirement (put it in `detail`), a ULN lab panel with a Gilbert syndrome
      exception, and prior-treatment exclusions. Replaces NCT06694454, which was
      suspended.
- [x] **NCT06706713** — BEBT-109, EGFR exon 20 insertion. Variant-specific requirement,
      a dense ULN lab panel, and two cohorts.

## Hard — the failures worth finding (~45 min each)

- [x] **NCT07198841** — IBI351 + cetuximab, untreated KRAS G12C. KRAS required plus
      several excluded drivers, ≥/≤ notation throughout, brain mets, ULN labs.
- [x] **NCT07483983** — ASP3082, KRAS G12D, multiple tumor types. Basket trial: decide
      what histology means across cohorts, plus prior-therapy rules that differ by
      cohort.
- [x] **NCT06092086** — first-line lorlatinib, ALK+, China. Over 10k characters, cohorts,
      a detailed brain-mets policy, and a dense lab panel.
- [x] **NCT06172478** — HER3-DXd basket, locally advanced or metastatic solid tumors.
      The longest (13k characters): many driver alterations, prior-line requirements per
      tumor type, multi-cohort.

## When all 18 are done

```bash
python eval/validate.py            # 0 errors, warnings reviewed
python eval/run_eval.py --runs 3   # about 60 parse calls
python eval/report.py
```

Then commit `eval/results.json` and add the headline numbers to the main README's
Results section.
