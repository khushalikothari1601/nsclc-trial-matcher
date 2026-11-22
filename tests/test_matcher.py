"""Unit tests for the deterministic matching engine.

The matcher is pure logic, so it gets exact-assertion unit tests -- not the eval
harness. These cover the rules most likely to break silently: the
UNKNOWN-not-FAIL invariant, boundary comparisons, biomarker direction, and the
brain-mets ordinal logic. Extend as you add evaluators.

Run:  pytest tests/ -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from schema import (  # noqa: E402
    BiomarkerGene,
    BiomarkerRequirement,
    BrainMetsStatus,
    LabRequirement,
    MatchStatus,
    PatientBiomarker,
    PatientBiomarkerStatus,
    PatientProfile,
    RequirementStatus,
    Sex,
    TrialCriteria,
)
from matcher import match_patient_to_trial  # noqa: E402


def _patient(**kw) -> PatientProfile:
    base = dict(age_years=60, sex=Sex.MALE)
    base.update(kw)
    return PatientProfile(**base)


def _trial(**kw) -> TrialCriteria:
    base = dict(nct_id="NCT_X")
    base.update(kw)
    return TrialCriteria(**base)


def _result_of(match, field_name: str) -> str:
    evaluations = match.passed_criteria + match.failed_criteria + match.unknown_criteria
    return next(e.result.value for e in evaluations if e.field_name == field_name)


# --- the core invariant: missing data is UNKNOWN, never FAIL ----------------

def test_missing_patient_data_never_fails():
    # trial requires ECOG <= 1, patient has no ECOG recorded
    result = match_patient_to_trial(_patient(ecog=None), _trial(ecog_max=1))
    assert result.match_status == MatchStatus.NEEDS_REVIEW
    assert not result.failed_criteria


def test_all_pass_is_likely_eligible():
    result = match_patient_to_trial(_patient(age_years=60, ecog=0), _trial(min_age_years=18, ecog_max=1))
    assert result.match_status == MatchStatus.LIKELY_ELIGIBLE


def test_explicit_contradiction_fails():
    result = match_patient_to_trial(_patient(ecog=3), _trial(ecog_max=1))
    assert result.match_status == MatchStatus.NOT_ELIGIBLE


# --- boundary conditions ----------------------------------------------------

@pytest.mark.parametrize("age,ok", [(18, True), (17, False), (75, True), (76, False)])
def test_age_boundaries(age, ok):
    result = match_patient_to_trial(_patient(age_years=age), _trial(min_age_years=18, max_age_years=75))
    failed = bool(result.failed_criteria)
    assert failed == (not ok)


@pytest.mark.parametrize("ecog,ok", [(0, True), (1, True), (2, False)])
def test_ecog_boundary(ecog, ok):
    result = match_patient_to_trial(_patient(ecog=ecog), _trial(ecog_max=1))
    assert (not result.failed_criteria) == ok


# --- biomarker direction (the eligibility-flipping field) -------------------

def test_biomarker_excluded_if_positive_blocks_positive_patient():
    result = match_patient_to_trial(
        _patient(biomarkers=[PatientBiomarker(gene=BiomarkerGene.EGFR, status=PatientBiomarkerStatus.POSITIVE)]),
        _trial(biomarkers=[BiomarkerRequirement(gene=BiomarkerGene.EGFR, status=RequirementStatus.EXCLUDED_IF_POSITIVE)]),
    )
    assert result.match_status == MatchStatus.NOT_ELIGIBLE


def test_biomarker_required_positive_needs_positive():
    result = match_patient_to_trial(
        _patient(biomarkers=[PatientBiomarker(gene=BiomarkerGene.ALK, status=PatientBiomarkerStatus.NEGATIVE)]),
        _trial(biomarkers=[BiomarkerRequirement(gene=BiomarkerGene.ALK, status=RequirementStatus.REQUIRED_POSITIVE)]),
    )
    assert result.match_status == MatchStatus.NOT_ELIGIBLE


def test_biomarker_unknown_status_is_review_not_fail():
    result = match_patient_to_trial(
        _patient(biomarkers=[PatientBiomarker(gene=BiomarkerGene.EGFR, status=PatientBiomarkerStatus.UNKNOWN)]),
        _trial(biomarkers=[BiomarkerRequirement(gene=BiomarkerGene.EGFR, status=RequirementStatus.REQUIRED_POSITIVE)]),
    )
    assert result.match_status == MatchStatus.NEEDS_REVIEW


# --- brain mets ordinal logic ----------------------------------------------

@pytest.mark.parametrize("patient_state,ok", [
    (BrainMetsStatus.NONE, True),
    (BrainMetsStatus.TREATED_STABLE, True),
    (BrainMetsStatus.UNTREATED, False),
    (BrainMetsStatus.SYMPTOMATIC, False),
])
def test_brain_mets_ordinal(patient_state, ok):
    # trial tolerates up to treated/stable
    result = match_patient_to_trial(
        _patient(brain_mets_status=patient_state),
        _trial(brain_mets_status=BrainMetsStatus.TREATED_STABLE),
    )
    assert (not result.failed_criteria) == ok


# --- unstructured text forces review ---------------------------------------

def test_unstructured_text_prevents_likely_eligible():
    result = match_patient_to_trial(
        _patient(age_years=60, ecog=0),
        _trial(min_age_years=18, ecog_max=1, manual_review_flag=True,
               unstructured_exclusion=["No active autoimmune disease"]),
    )
    assert result.match_status == MatchStatus.NEEDS_REVIEW


# --- free-text trial terms: interpret what's clear, never fail on the rest -----

@pytest.mark.parametrize("term,histology,expected", [
    ("NSCLC (histologically confirmed)", "adenocarcinoma", "pass"),
    ("non-small cell lung cancer (NSCLC), histologically confirmed", "squamous", "pass"),
    ("non-squamous NSCLC", "adenocarcinoma", "pass"),
    ("non-squamous NSCLC", "large_cell", "pass"),
    ("non-squamous NSCLC", "squamous", "fail"),
    ("squamous NSCLC", "adenocarcinoma", "fail"),
    ("squamous NSCLC", "squamous", "pass"),
    ("squamous and non-squamous non-small cell lung cancer", "squamous", "pass"),
    ("Histologic evidence of a malignant solid tumor", "adenocarcinoma", "unknown"),
    ("melanoma", "adenocarcinoma", "unknown"),
])
def test_histology_inclusion_terms(term, histology, expected):
    result = match_patient_to_trial(_patient(histology=histology), _trial(histology_included=[term]))
    assert _result_of(result, "histology") == expected


@pytest.mark.parametrize("term,histology,expected", [
    ("squamous NSCLC (only nonsquamous NSCLC eligible)", "squamous", "fail"),
    ("squamous NSCLC (only nonsquamous NSCLC eligible)", "adenocarcinoma", "pass"),
    ("small cell lung cancer", "adenocarcinoma", "pass"),
    ("Mixed small cell and non-small cell lung cancer histology", "adenocarcinoma", "unknown"),
    ("NSCLC with mixed small cell components", "squamous", "unknown"),
    ("squamous cell skin cancer", "squamous", "unknown"),
    ("NSCLC", "adenocarcinoma", "unknown"),
])
def test_histology_exclusion_terms(term, histology, expected):
    result = match_patient_to_trial(_patient(histology=histology), _trial(histology_excluded=[term]))
    assert _result_of(result, "histology") == expected


def test_other_cancer_cohort_does_not_match_adenocarcinoma():
    result = match_patient_to_trial(
        _patient(histology="adenocarcinoma"),
        _trial(histology_included=["pancreatic ductal adenocarcinoma (PDAC), histologically confirmed (cohort 2)"]),
    )
    assert _result_of(result, "histology") == "unknown"


@pytest.mark.parametrize("value", ["unknown", "Unknown", "other"])
def test_uninformative_histology_is_not_a_failure(value):
    result = match_patient_to_trial(_patient(histology=value), _trial(histology_included=["adenocarcinoma"]))
    assert _result_of(result, "histology") == "unknown"


def test_unknown_stage_is_not_a_failure():
    result = match_patient_to_trial(_patient(stage="unknown"), _trial(stage_included=["IV"]))
    assert _result_of(result, "stage") == "unknown"


@pytest.mark.parametrize("term,stage,expected", [
    ("IV", "metastatic", "pass"),
    ("metastatic", "IV", "pass"),
    ("IV", "IVA", "pass"),
    ("III", "IIIB", "pass"),
    ("IIIB", "III", "unknown"),
    ("IIIB", "IIIA", "fail"),
    ("IIIB", "IV", "fail"),
    ("IIIB (not amenable for multimodality treatment)", "IIIB", "pass"),
    ("T3-4N2 IIIB", "IIIB", "pass"),
    ("3", "IIIB", "pass"),
    ("locally advanced", "IIIB", "unknown"),
    ("unresectable", "IV", "unknown"),
    ("IIIB-IV", "IV", "unknown"),
])
def test_stage_inclusion_terms(term, stage, expected):
    result = match_patient_to_trial(_patient(stage=stage), _trial(stage_included=[term]))
    assert _result_of(result, "stage") == expected


def test_stage_included_any_term_matching_passes():
    result = match_patient_to_trial(_patient(stage="IV"), _trial(stage_included=["locally advanced", "metastatic"]))
    assert _result_of(result, "stage") == "pass"


@pytest.mark.parametrize("term,stage,expected", [("M1 disease", "IV", "fail"), ("metastatic NSCLC", "IVA", "fail"), ("M1", "IV", "fail"), ("M1", "IIIA", "pass"), ("TxN3", "IIIB", "unknown")])
def test_stage_exclusion_terms(term, stage, expected):
    result = match_patient_to_trial(_patient(stage=stage), _trial(stage_excluded=[term]))
    assert _result_of(result, "stage") == expected


# --- labs: only compare like with like ---------------------------------------

def test_uln_relative_threshold_is_not_compared_to_absolute_value():
    result = match_patient_to_trial(
        _patient(organ_function_labs={"Bilirubin": 1.0}),
        _trial(organ_function_labs=[LabRequirement(lab_name="bilirubin", operator="<=", value=0.5, unit="x ULN")]),
    )
    assert _result_of(result, "lab_bilirubin") == "unknown"


def test_other_unit_threshold_is_not_compared():
    result = match_patient_to_trial(
        _patient(organ_function_labs={"ANC": 1.5}),
        _trial(organ_function_labs=[LabRequirement(lab_name="ANC", operator=">=", value=1500, unit="/microL")]),
    )
    assert _result_of(result, "lab_anc") == "unknown"


@pytest.mark.parametrize("trial_name,trial_unit", [
    ("ANC", "x10^9/L"),
    ("absolute neutrophil count", "10^9/L"),
    ("Neutrophils", "x 10^9/L"),
])
def test_lab_names_and_units_align(trial_name, trial_unit):
    result = match_patient_to_trial(
        _patient(organ_function_labs={"ANC": 1.0}),
        _trial(organ_function_labs=[LabRequirement(lab_name=trial_name, operator=">=", value=1.5, unit=trial_unit)]),
    )
    assert _result_of(result, "lab_anc") == "fail"


# --- prior treatment strings carry no drug-class knowledge -------------------

def _history(*drugs):
    return dict(prior_treatment_history_provided=True, prior_treatments_received=list(drugs))


def test_required_class_vs_specific_drug_is_unknown():
    result = match_patient_to_trial(
        _patient(**_history("carboplatin")), _trial(required_prior_treatments=["platinum-based chemotherapy"])
    )
    assert result.match_status == MatchStatus.NEEDS_REVIEW


def test_required_treatment_with_empty_history_fails():
    result = match_patient_to_trial(_patient(**_history()), _trial(required_prior_treatments=["osimertinib"]))
    assert result.match_status == MatchStatus.NOT_ELIGIBLE


def test_excluded_class_vs_specific_drug_is_not_a_pass():
    result = match_patient_to_trial(
        _patient(**_history("sotorasib")), _trial(excluded_prior_treatments=["any KRAS G12C inhibitor"])
    )
    assert result.match_status == MatchStatus.NEEDS_REVIEW


def test_excluded_treatment_string_match_fails():
    result = match_patient_to_trial(_patient(**_history("osimertinib")), _trial(excluded_prior_treatments=["osimertinib"]))
    assert result.match_status == MatchStatus.NOT_ELIGIBLE


# --- biomarker variants and remaining edge cases -----------------------------

def _egfr(status, detail=None):
    return PatientBiomarker(gene=BiomarkerGene.EGFR, status=status, detail=detail)


def test_required_variant_with_unspecified_positive_needs_review():
    result = match_patient_to_trial(
        _patient(biomarkers=[_egfr(PatientBiomarkerStatus.POSITIVE)]),
        _trial(biomarkers=[BiomarkerRequirement(gene=BiomarkerGene.EGFR, status=RequirementStatus.REQUIRED_POSITIVE,
                                                detail="exon 20 insertion")]),
    )
    assert result.match_status == MatchStatus.NEEDS_REVIEW


def test_excluded_variant_with_unspecified_positive_needs_review():
    result = match_patient_to_trial(
        _patient(biomarkers=[_egfr(PatientBiomarkerStatus.POSITIVE)]),
        _trial(biomarkers=[BiomarkerRequirement(gene=BiomarkerGene.EGFR, status=RequirementStatus.EXCLUDED_IF_POSITIVE,
                                                detail="T790M")]),
    )
    assert result.match_status == MatchStatus.NEEDS_REVIEW


def test_matching_variant_passes():
    result = match_patient_to_trial(
        _patient(biomarkers=[_egfr(PatientBiomarkerStatus.POSITIVE, "exon 19 deletion")]),
        _trial(biomarkers=[BiomarkerRequirement(gene=BiomarkerGene.EGFR, status=RequirementStatus.REQUIRED_POSITIVE,
                                                detail="exon 19 deletion")]),
    )
    assert result.match_status == MatchStatus.LIKELY_ELIGIBLE


def test_negative_patient_still_fails_required_variant():
    result = match_patient_to_trial(
        _patient(biomarkers=[_egfr(PatientBiomarkerStatus.NEGATIVE)]),
        _trial(biomarkers=[BiomarkerRequirement(gene=BiomarkerGene.EGFR, status=RequirementStatus.REQUIRED_POSITIVE,
                                                detail="exon 19 deletion")]),
    )
    assert result.match_status == MatchStatus.NOT_ELIGIBLE


def test_trial_brain_mets_unknown_does_not_crash():
    result = match_patient_to_trial(
        _patient(brain_mets_status=BrainMetsStatus.UNTREATED), _trial(brain_mets_status=BrainMetsStatus.UNKNOWN)
    )
    assert result.match_status == MatchStatus.LIKELY_ELIGIBLE


def test_measurable_disease_not_required_is_no_constraint():
    result = match_patient_to_trial(_patient(measurable_disease=None), _trial(measurable_disease_required=False))
    assert result.match_status == MatchStatus.LIKELY_ELIGIBLE
