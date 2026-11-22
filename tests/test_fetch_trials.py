"""The parser must receive the age/sex limits, which ClinicalTrials.gov keeps outside
the free text. Gold labels assume it does."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fetch_trials import _extract_trial, format_eligibility_text  # noqa: E402


def _study(**eligibility):
    return {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001", "briefTitle": "A trial"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "eligibilityModule": eligibility,
        }
    }


def test_header_carries_structured_limits():
    text = format_eligibility_text(
        {"minimumAge": "18 Years", "maximumAge": "75 Years", "sex": "ALL", "healthyVolunteers": False,
         "eligibilityCriteria": "Inclusion Criteria:\n* ECOG 0-1"}
    )
    assert text == (
        "Ages Eligible for Study: 18 Years to 75 Years.\nSexes Eligible for Study: ALL.\n"
        "Accepts Healthy Volunteers: No.\n\nInclusion Criteria:\n* ECOG 0-1"
    )


def test_no_structured_fields_leaves_text_alone():
    assert format_eligibility_text({"eligibilityCriteria": " Inclusion Criteria: x "}) == "Inclusion Criteria: x"


def test_pipeline_input_includes_header():
    trial = _extract_trial(_study(minimumAge="18 Years", eligibilityCriteria="Inclusion Criteria: x"))
    assert "18 Years" in trial.eligibility_criteria


def test_trial_without_criteria_is_dropped():
    assert _extract_trial(_study(minimumAge="18 Years")) is None
