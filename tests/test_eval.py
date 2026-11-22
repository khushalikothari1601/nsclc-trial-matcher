"""Tests for the eval harness itself: a scorer that misreports is worse than no scorer.
No model calls -- run_eval's parse step is replaced with a stub.

Run:  pytest tests/ -v
"""

import json
import os
import shutil
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

import pytest  # noqa: E402

import report  # noqa: E402
import run_eval  # noqa: E402
from schema import (  # noqa: E402
    BiomarkerGene,
    BiomarkerRequirement,
    LabRequirement,
    RequirementStatus,
    TrialCriteria,
)
from scorer import Outcome, Severity, score_trial  # noqa: E402

FIXTURE_GOLD = os.path.join(os.path.dirname(__file__), "fixtures", "gold")


def _criteria(**kw) -> TrialCriteria:
    return TrialCriteria(nct_id="NCT_X", **kw)


def _field(gold: TrialCriteria, pred: TrialCriteria, name: str):
    return next(f for f in score_trial(gold, pred).field_scores if f.field_name == name)


def _biomarker(status, gene=BiomarkerGene.EGFR):
    return BiomarkerRequirement(gene=gene, status=status)


# --- accuracy ---------------------------------------------------------------

def test_identical_parse_scores_perfect():
    gold = _criteria(ecog_max=1, stage_included=["IV"], unstructured_exclusion=["Active autoimmune disease"])
    score = score_trial(gold, gold)
    assert score.accuracy == 1.0
    assert score.populated_accuracy == 1.0


def test_populated_accuracy_ignores_fields_neither_side_mentions():
    score = score_trial(_criteria(ecog_max=1), _criteria(ecog_max=2))
    assert score.accuracy > 0.9  # the headline number barely notices
    assert score.populated_accuracy == 0.0


# --- scalars ----------------------------------------------------------------

@pytest.mark.parametrize("gold,pred,outcome", [
    (dict(ecog_max=1), dict(), Outcome.MISSED),
    (dict(), dict(ecog_max=1), Outcome.HALLUCINATED),
    (dict(ecog_max=1), dict(ecog_max=2), Outcome.WRONG_VALUE),
    (dict(pdl1_tps_min=0), dict(), Outcome.MISSED),  # 0 is a value, not "not mentioned"
])
def test_scalar_error_classes(gold, pred, outcome):
    field_name = next(iter(gold or pred))
    assert _field(_criteria(**gold), _criteria(**pred), field_name).outcome == outcome


def test_unstated_sex_equals_all():
    field = _field(_criteria(sex="all"), _criteria(), "sex")
    assert field.outcome == Outcome.CORRECT
    assert not field.populated


# --- sets -------------------------------------------------------------------

@pytest.mark.parametrize("field_name,gold,pred", [
    ("stage_included", ["IV"], ["metastatic"]),
    ("stage_included", ["IIIB", "IV"], ["Stage IV", "IIIB"]),
    ("histology_excluded", ["squamous"], ["squamous NSCLC"]),
])
def test_equivalent_terms_are_not_errors(field_name, gold, pred):
    assert _field(_criteria(**{field_name: gold}), _criteria(**{field_name: pred}), field_name).correct


def test_dropped_exclusion_is_critical():
    field = _field(_criteria(stage_excluded=["IIIA"]), _criteria(), "stage_excluded")
    assert field.outcome == Outcome.MISSED
    assert field.severity == Severity.CRITICAL


def test_prior_treatment_clauses_align_fuzzily():
    gold = _criteria(excluded_prior_treatments=["any anti-PD-1 or anti-PD-L1 antibody"])
    pred = _criteria(excluded_prior_treatments=["prior anti-PD-1 / anti-PD-L1 antibody therapy"])
    assert _field(gold, pred, "excluded_prior_treatments").correct


@pytest.mark.parametrize("field_name,gold,pred,correct", [
    ("histology_excluded", ["mixed tumors containing small cell elements"], ["mixed tumors with small cell elements"], True),
    ("histology_excluded", ["squamous"], ["non-squamous"], False),
    ("histology_excluded", ["small cell lung cancer", "Pancoast tumor"], ["SCLC", "sarcomatoid carcinoma"], False),
    ("stage_excluded", ["M1 disease"], ["IV"], True),
    ("stage_excluded", ["IIIA"], ["IIIB"], False),
    ("stage_excluded", ["N3 disease", "M1 disease"], ["TxN3", "Stage IV"], False),
])
def test_free_text_list_items_align_by_wording(field_name, gold, pred, correct):
    assert _field(_criteria(**{field_name: gold}), _criteria(**{field_name: pred}), field_name).correct is correct


def test_list_scores_count_unmatched_items():
    gold = _criteria(unstructured_exclusion=["Active autoimmune disease", "Prior organ transplant", "Active hepatitis B"])
    pred = _criteria(unstructured_exclusion=["Active autoimmune disease", "QTcF above 470 ms"])
    field = _field(gold, pred, "unstructured_exclusion")
    assert (field.n_missing, field.n_extra) == (2, 1)
    assert score_trial(gold, pred).unstructured_items() == (3, 1, 2, 1)


# --- biomarkers -------------------------------------------------------------

def test_flipped_biomarker_direction_is_critical():
    gold = _criteria(biomarkers=[_biomarker(RequirementStatus.REQUIRED_POSITIVE)])
    pred = _criteria(biomarkers=[_biomarker(RequirementStatus.EXCLUDED_IF_POSITIVE)])
    score = score_trial(gold, pred)
    assert [f.field_name for f in score.critical_errors()] == ["biomarker_EGFR"]


def test_two_requirements_for_one_gene_are_both_scored():
    both = [_biomarker(RequirementStatus.REQUIRED_POSITIVE), _biomarker(RequirementStatus.EXCLUDED_IF_POSITIVE)]
    gold = _criteria(biomarkers=both)
    assert _field(gold, gold, "biomarker_EGFR").correct
    dropped_one = _criteria(biomarkers=both[:1])
    assert _field(gold, dropped_one, "biomarker_EGFR").outcome == Outcome.WRONG_VALUE


# --- labs -------------------------------------------------------------------

def test_lab_aliases_align():
    gold = _criteria(organ_function_labs=[LabRequirement(lab_name="ANC", operator=">=", value=1.5)])
    pred = _criteria(organ_function_labs=[LabRequirement(lab_name="absolute neutrophil count", operator=">=", value=1.5)])
    assert _field(gold, pred, "lab_anc").correct


@pytest.mark.parametrize("gold_name,pred_name", [("TBIL", "total bilirubin"), ("Cr", "serum creatinine")])
def test_lab_abbreviations_align(gold_name, pred_name):
    gold = _criteria(organ_function_labs=[LabRequirement(lab_name=gold_name, operator="<=", value=1.5, unit="x ULN")])
    pred = _criteria(organ_function_labs=[LabRequirement(lab_name=pred_name, operator="<=", value=1.5, unit="x ULN")])
    assert score_trial(gold, pred).populated_accuracy == 1.0


def test_lab_wrong_threshold():
    gold = _criteria(organ_function_labs=[LabRequirement(lab_name="ANC", operator=">=", value=1.5)])
    pred = _criteria(organ_function_labs=[LabRequirement(lab_name="ANC", operator=">=", value=1.0)])
    assert _field(gold, pred, "lab_anc").outcome == Outcome.WRONG_VALUE


# --- unstructured buckets ---------------------------------------------------

def test_near_verbatim_items_align():
    gold = _criteria(unstructured_exclusion=["Active autoimmune disease or history of autoimmune disease"])
    pred = _criteria(unstructured_exclusion=["History of active autoimmune disease"])
    assert _field(gold, pred, "unstructured_exclusion").correct


def test_merged_items_align():
    gold = _criteria(unstructured_exclusion=["Pregnant", "Breast-feeding"])
    pred = _criteria(unstructured_exclusion=["Pregnant or breast-feeding women"])
    assert _field(gold, pred, "unstructured_exclusion").correct


def test_over_structuring_is_moderate():
    gold = _criteria(unstructured_exclusion=["Active autoimmune disease", "Prior organ transplant"])
    pred = _criteria(unstructured_exclusion=["Active autoimmune disease"])
    field = _field(gold, pred, "unstructured_exclusion")
    assert field.outcome == Outcome.MISSED
    assert field.severity == Severity.MODERATE


def test_over_flagging_is_minor():
    gold = _criteria(unstructured_exclusion=["Active autoimmune disease"])
    pred = _criteria(unstructured_exclusion=["Active autoimmune disease", "ECOG performance status above 1"])
    field = _field(gold, pred, "unstructured_exclusion")
    assert field.outcome == Outcome.HALLUCINATED
    assert field.severity == Severity.MINOR


def test_unrelated_items_do_not_align():
    gold = _criteria(unstructured_exclusion=["Active autoimmune disease"])
    pred = _criteria(unstructured_exclusion=["Prior organ transplant"])
    assert _field(gold, pred, "unstructured_exclusion").outcome == Outcome.WRONG_VALUE


# --- runner + report, end to end --------------------------------------------

def _perfect_parse(nct_id, title, status, raw, model=None):
    return next(c.gold for c in run_eval.load_gold(FIXTURE_GOLD) if c.nct_id == nct_id)


def test_end_to_end_run_and_report(tmp_path, capsys, monkeypatch):
    out = tmp_path / "results.json"
    run_eval.run(FIXTURE_GOLD, 2, str(out), parse=_perfect_parse)

    data = json.loads(out.read_text())
    assert data["n_cases"] == 1
    assert len(data["runs"]) == 2
    assert all(r["populated_accuracy"] == 1.0 for r in data["runs"])

    monkeypatch.setattr(sys, "argv", ["report.py", "--results", str(out)])
    report.main()
    printed = capsys.readouterr().out
    assert "Accuracy, populated fields: 100.0%" in printed
    assert "smoke test" in printed


def test_parse_failure_is_recorded_and_report_survives(tmp_path, capsys, monkeypatch):
    def broken(*args):
        raise RuntimeError("model refused")

    out = tmp_path / "results.json"
    run_eval.run(FIXTURE_GOLD, 1, str(out), parse=broken)
    assert json.loads(out.read_text())["runs"][0]["parse_error"] == "model refused"

    monkeypatch.setattr(sys, "argv", ["report.py", "--results", str(out)])
    report.main()
    assert "1 parse errors" in capsys.readouterr().out


def test_partial_results_survive_an_interrupted_run(tmp_path):
    gold_dir = tmp_path / "gold"
    shutil.copytree(FIXTURE_GOLD, gold_dir)
    blob = json.loads((gold_dir / "NCT_TEST01.json").read_text())
    blob["nct_id"] = "NCT_TEST02"
    (gold_dir / "NCT_TEST02.json").write_text(json.dumps(blob))

    calls = []

    def interrupted(nct_id, title, status, raw, model=None):
        calls.append(nct_id)
        if len(calls) > 1:
            raise KeyboardInterrupt
        return TrialCriteria(nct_id=nct_id)

    out = tmp_path / "results.json"
    with pytest.raises(KeyboardInterrupt):
        run_eval.run(str(gold_dir), 1, str(out), parse=interrupted)
    assert len(json.loads(out.read_text())["runs"]) == 1


def test_predictions_are_saved_and_rescore_matches(tmp_path):
    import rescore

    out = tmp_path / "results.json"
    run_eval.run(FIXTURE_GOLD, 1, str(out), parse=_perfect_parse)
    first = json.loads(out.read_text())
    assert first["runs"][0]["prediction"]["nct_id"] == "NCT_TEST01"

    rescored = tmp_path / "rescored.json"
    rescore.rescore(str(out), str(rescored), FIXTURE_GOLD)
    again = json.loads(rescored.read_text())
    assert again["runs"][0]["field_scores"] == first["runs"][0]["field_scores"]


def test_unlabeled_gold_file_is_skipped(tmp_path):
    blob = {
        "_raw_eligibility_text": "Inclusion Criteria: ...",
        "nct_id": "NCT00000001",
        "trial_title": "Prefilled only",
        "overall_status": "RECRUITING",
        "min_age_years": 18,
        "sex": "all",
        "ecog_max": None,
        "biomarkers": [],
        "manual_review_flag": False,
    }
    (tmp_path / "NCT00000001.json").write_text(json.dumps(blob))
    assert run_eval.load_gold(str(tmp_path)) == []


def test_every_labeled_gold_file_loads():
    cases = run_eval.load_gold(os.path.join(ROOT, "eval", "gold"))
    assert {c.nct_id for c in cases} >= {"NCT04973293", "NCT05393466"}
