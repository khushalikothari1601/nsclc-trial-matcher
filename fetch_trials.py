"""Thin client for the ClinicalTrials.gov API v2 (public, no auth required).

Docs: https://clinicaltrials.gov/data-api/api
"""

from dataclasses import dataclass

import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_CONDITION = "non small cell lung cancer"
PAGE_SIZE = 100


@dataclass
class RawTrial:
    nct_id: str
    brief_title: str
    overall_status: str
    eligibility_criteria: str


def format_eligibility_text(eligibility: dict) -> str:
    """The text the parser sees: the criteria, preceded by the age/sex limits that
    ClinicalTrials.gov stores in structured fields rather than in the free text.

    Gold files are built with this too (eval/label.py), so the eval scores the same
    input the pipeline parses."""

    lines = []
    minimum_age, maximum_age = eligibility.get("minimumAge"), eligibility.get("maximumAge")
    if minimum_age or maximum_age:
        lines.append(f"Ages Eligible for Study: {minimum_age or 'N/A'} to {maximum_age or 'no upper limit'}.")
    if eligibility.get("sex"):
        lines.append(f"Sexes Eligible for Study: {eligibility['sex']}.")
    healthy = eligibility.get("healthyVolunteers")
    if healthy is not None:
        lines.append(f"Accepts Healthy Volunteers: {'Yes' if healthy else 'No'}.")

    criteria = eligibility.get("eligibilityCriteria", "").strip()
    header = "\n".join(lines)
    return f"{header}\n\n{criteria}" if header else criteria


def _extract_trial(study: dict) -> RawTrial | None:
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    status = protocol.get("statusModule", {})

    nct_id = identification.get("nctId")
    if not nct_id or not eligibility.get("eligibilityCriteria"):
        return None

    return RawTrial(
        nct_id=nct_id,
        brief_title=identification.get("briefTitle", ""),
        overall_status=status.get("overallStatus", ""),
        eligibility_criteria=format_eligibility_text(eligibility),
    )


def fetch_nsclc_trials(
    max_trials: int = 50,
    condition: str = DEFAULT_CONDITION,
    status: str = "RECRUITING",
) -> list[RawTrial]:
    """Fetch up to max_trials NSCLC trials from ClinicalTrials.gov, paginating via
    nextPageToken until either max_trials is reached or results are exhausted."""

    trials: list[RawTrial] = []
    page_token: str | None = None

    while len(trials) < max_trials:
        params = {
            "query.cond": condition,
            "filter.overallStatus": status,
            "pageSize": min(PAGE_SIZE, max_trials - len(trials)),
            "format": "json",
        }
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        for study in data.get("studies", []):
            trial = _extract_trial(study)
            if trial is not None:
                trials.append(trial)
            if len(trials) >= max_trials:
                break

        page_token = data.get("nextPageToken")
        if not page_token or not data.get("studies"):
            break

    return trials
