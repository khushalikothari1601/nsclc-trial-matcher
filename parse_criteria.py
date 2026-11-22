"""Use Claude to parse a single trial's free-text eligibility criteria into the
structured TrialCriteriaExtracted schema. This is the only module that calls the
Anthropic API -- matching itself (matcher.py) is plain deterministic Python."""

import os

import anthropic

from schema import TrialCriteria, TrialCriteriaExtracted

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")

SYSTEM_PROMPT = """You are extracting structured eligibility data from a single \
ClinicalTrials.gov NSCLC (non-small cell lung cancer) trial's free-text eligibility \
criteria.

Rules:
- Only extract what is explicitly stated. Do not infer or guess a value that isn't there.
- Use the exact enum values given in the schema.
- If a criterion doesn't map cleanly onto a schema field, put the criterion text \
(near-verbatim) into unstructured_inclusion or unstructured_exclusion instead of \
forcing it into a field or dropping it.
- Distinguish inclusion vs exclusion criteria using the trial's own headers/structure.
- Leave a field null/empty if the trial text doesn't address it -- do not guess a default.
- Set parse_confidence to 'low' if the text is ambiguous, contradictory, or unusually \
structured.

Biomarkers: use only the genes the schema lists, and KRAS_G12C means only the G12C \
variant. If the trial names a gene or variant the schema can't express (for example \
KRAS G12D or NRAS), put the criterion in the unstructured lists. Never map it to the \
nearest listed value.

Multi-cohort trials: put a criterion in a structured field only if it applies to every \
participant. A criterion that applies to one cohort, phase or tumor type goes in the \
unstructured lists, prefixed with the cohort it applies to.

brain_mets_status is the most permissive brain-metastasis state the trial accepts. If \
only active, untreated, symptomatic or unstable metastases are excluded, use \
treated_stable. Use none only when all brain metastases are excluded. Leave it null if \
brain metastases aren't addressed.

Prior treatment -- put each rule in exactly one place:
- "No prior systemic therapy" / treatment-naive (optionally "for advanced disease"): \
treatment_naive_required = true. Do not also list the classes it implies in \
excluded_prior_treatments or set prior_therapy_lines to 0. Put any setting exception \
(e.g. prior adjuvant therapy allowed) in unstructured_inclusion.
- A count of prior lines ("at least 2 prior lines", "1-2 prior lines"): \
prior_therapy_lines_min / prior_therapy_lines_max, and leave treatment_naive_required null.
- A specific prior drug or class that is required: required_prior_treatments, and leave \
treatment_naive_required null. Use treatment_naive_required = false only when prior \
therapy is required with no count and no named treatment.
- A requirement that applies only to one cohort or condition: unstructured_inclusion.
- A specific prior drug or class that is excluded, with no time window: \
excluded_prior_treatments, one short entry each.
- Washout or time-window rules ("within 4 weeks"), concomitant medications, vaccines and \
participation in other trials: unstructured_exclusion, never excluded_prior_treatments.
"""


def parse_trial_criteria(
    nct_id: str,
    trial_title: str,
    overall_status: str,
    eligibility_text: str,
    model: str = DEFAULT_MODEL,
) -> TrialCriteria:
    client = anthropic.Anthropic()

    response = client.messages.parse(
        model=model,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"NCT ID: {nct_id}\n\nEligibility criteria:\n{eligibility_text}",
            }
        ],
        output_format=TrialCriteriaExtracted,
    )

    extracted = response.parsed_output

    return TrialCriteria.from_extracted(
        nct_id=nct_id,
        trial_title=trial_title,
        overall_status=overall_status,
        extracted=extracted,
    )
