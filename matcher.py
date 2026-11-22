"""Deterministic, pure-Python matching engine. No I/O, no LLM calls -- this module
only compares a PatientProfile against a TrialCriteria and produces a MatchResult, so
results are explainable and reproducible for a given (patient, trial-criteria) pair.

Core rule: a criterion is FAIL only when the patient's *known* value explicitly
contradicts the trial's requirement. Missing/unset patient data is always UNKNOWN,
never FAIL -- this is what keeps "needs review" from silently becoming a false
negative (or the unstructured fallback bucket from becoming a false "Likely Eligible").

The same rule covers the free text the LLM leaves in stage, histology, prior-treatment
and lab fields: a term the matcher can't confidently interpret is UNKNOWN, not a
contradiction.
"""

import re

from schema import (
    BrainMetsStatus,
    CriterionEvaluation,
    CriterionResultEnum,
    MatchResult,
    MatchStatus,
    PatientBiomarkerStatus,
    PatientProfile,
    RequirementStatus,
    Sex,
    TrialCriteria,
)

PASS = CriterionResultEnum.PASS
FAIL = CriterionResultEnum.FAIL
UNKNOWN = CriterionResultEnum.UNKNOWN

# Exact-string stage synonyms. Terms like "locally advanced" or "unresectable" are
# deliberately absent: they don't name a stage, so they must stay uninterpretable
# (UNKNOWN) rather than being mapped onto one.
STAGE_SYNONYMS = {
    "metastatic": "iv",
    "m1": "iv",
    "stage iv": "iv",
    "stage 4": "iv",
    "stage 3b": "iiib",
    "stage 3": "iii",
}

_STAGE_TOKEN = re.compile(r"\b(?:stage\s*)?(iv|iii|ii|i|[1-4])([abc]?)\b")
_ARABIC_STAGE = {"1": "i", "2": "ii", "3": "iii", "4": "iv"}

# Canonical histology terms, checked in order: specific subtypes before the generic
# "nsclc", and "non-small cell" before "small cell". Trial criteria arrive as free
# text ("NSCLC (histologically confirmed)"), so this is substring matching.
HISTOLOGY_PATTERNS = [
    ("large_cell_neuroendocrine", re.compile(r"large[\s_-]cell neuroendocrine")),
    ("non_squamous", re.compile(r"non[\s-]?squamous")),
    ("squamous", re.compile(r"squamous|\bscc\b")),
    ("adenocarcinoma", re.compile(r"adenocarcinoma")),
    ("large_cell", re.compile(r"large[\s_-]cell")),
    ("nsclc", re.compile(r"non[\s-]small[\s-]cell|\bnsclc\b")),
    ("small_cell", re.compile(r"small[\s-]cell|\bsclc\b")),
]

# Text naming a mixed tumor or another organ site uses lung-histology words without
# describing a lung histology ("squamous cell skin cancer", "pancreatic ductal
# adenocarcinoma"), so it stays uninterpretable rather than matching the patient.
HISTOLOGY_NOT_APPLICABLE = re.compile(
    r"\bmixed\b|skin|head and neck|esophag|pancrea|pdac|bladder|urothelial|breast|colorectal"
    r"|ovarian|cervical|endometrial|gastric|prostate|renal|hepatocellular|thyroid"
)
_BOTH_SQUAMOUS_GROUPS = re.compile(r"(?<!non)(?<!non[\s-])squamous.*non[\s-]?squamous|non[\s-]?squamous.*(?<!non)(?<!non[\s-])squamous")

# What each patient histology satisfies, beyond its own name.
HISTOLOGY_PARENTS = {
    "adenocarcinoma": {"non_squamous", "nsclc"},
    "large_cell": {"non_squamous", "nsclc"},
    "large_cell_neuroendocrine": {"non_squamous", "nsclc", "large_cell"},
    "squamous": {"nsclc"},
    "non_squamous": {"nsclc"},
    "nsclc": set(),
}

# Patient values that carry no information and so can never contradict a criterion.
UNINFORMATIVE_PATIENT_VALUES = {"unknown", "other", "not specified", "n/a"}

# Ordinal permissiveness of a brain-mets stance: how "bad" a state the trial tolerates.
# criteria.brain_mets_status is read as the most permissive patient state the trial
# accepts -- e.g. TREATED_STABLE means the trial allows both no-mets and
# treated/stable mets, but not untreated or symptomatic ones.
BRAIN_METS_ORDER = {
    BrainMetsStatus.NONE: 0,
    BrainMetsStatus.TREATED_STABLE: 1,
    BrainMetsStatus.UNTREATED: 2,
    BrainMetsStatus.SYMPTOMATIC: 3,
}

LAB_OPERATORS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}


LAB_ALIASES = {
    "absolute_neutrophil_count": "anc",
    "neutrophils": "anc",
    "neutrophil_count": "anc",
    "platelet": "platelets",
    "platelet_count": "platelets",
    "plt": "platelets",
    "hgb": "hemoglobin",
    "hb": "hemoglobin",
    "total_bilirubin": "bilirubin",
    "tbil": "bilirubin",
    "t_bil": "bilirubin",
    "cr": "creatinine",
    "scr": "creatinine",
    "serum_creatinine": "creatinine",
    "blood_urea_nitrogen": "bun",
    "serum_bilirubin": "bilirubin",
    "crcl": "creatinine_clearance",
    "calculated_creatinine_clearance": "creatinine_clearance",
    "aspartate_aminotransferase": "ast",
    "sgot": "ast",
    "alanine_aminotransferase": "alt",
    "sgpt": "alt",
}

# Units the patient form collects each lab in (normalized by _norm_unit). A trial
# threshold in any other unit is not comparable, so it evaluates to UNKNOWN.
PATIENT_LAB_UNITS = {
    "anc": {"10^9/l"},
    "platelets": {"10^9/l"},
    "hemoglobin": {"g/dl"},
    "creatinine_clearance": {"ml/min", "ml/min/1.73m^2"},
    "bilirubin": {"mg/dl"},
    "ast": {"u/l"},
    "alt": {"u/l"},
}


def _norm(s: str) -> str:
    return s.strip().lower()


def _norm_lab(s: str) -> str:
    key = _norm(s).replace(" ", "_").replace("-", "_")
    return LAB_ALIASES.get(key, key)


def _norm_unit(s: str) -> str:
    return re.sub(r"^x", "", _norm(s).replace(" ", "").replace("\u00d7", "x"))


def _lab_unit_comparable(key: str, unit: str | None) -> bool:
    return bool(unit) and _norm_unit(unit) in PATIENT_LAB_UNITS.get(key, set())


def _parse_stage(s: str) -> tuple[str, str] | None:
    """Return (major, substage) for text naming a stage, else None."""
    n = re.sub(r"\b(disease|nsclc)\b", " ", _norm(s))
    n = " ".join(n.split())
    n = STAGE_SYNONYMS.get(n, n)
    tokens = set(_STAGE_TOKEN.findall(n))
    if len(tokens) != 1:  # none, or a range like "IIIB-IV" that can't be read as one stage
        return None
    major, sub = tokens.pop()
    return _ARABIC_STAGE.get(major, major), sub


def _norm_stage(s: str) -> str:
    """Canonical stage ('iiib') if s names one, otherwise the normalized text."""
    parsed = _parse_stage(s)
    return "".join(parsed) if parsed else _norm(s)


def _norm_histology(s: str) -> str:
    """Canonical histology term if s names one, otherwise the normalized text."""
    n = _norm(s)
    if HISTOLOGY_NOT_APPLICABLE.search(n):
        return n
    # Parentheticals qualify the term ("squamous NSCLC (only nonsquamous eligible)").
    core = re.sub(r"\(.*?\)", "", n) or n
    if _BOTH_SQUAMOUS_GROUPS.search(core):
        return "nsclc"
    for canonical, pattern in HISTOLOGY_PATTERNS:
        if pattern.search(core):
            return canonical
    return n


def _is_known_histology(term: str) -> bool:
    return term in {name for name, _ in HISTOLOGY_PATTERNS}


def _histology_matches(patient: str, term: str) -> bool:
    return patient == term or term in HISTOLOGY_PARENTS.get(patient, set())


def _stage_relation(patient: tuple[str, str], term: tuple[str, str]) -> CriterionResultEnum:
    """PASS if the patient's stage falls within term, FAIL if it can't, UNKNOWN if
    the term is more specific than what's known about the patient (III vs IIIB)."""
    if patient[0] != term[0]:
        return FAIL
    if not term[1] or patient[1] == term[1]:
        return PASS
    return UNKNOWN if not patient[1] else FAIL


def _combine_inclusion(results: list[CriterionResultEnum]) -> CriterionResultEnum:
    """Any term matching passes; FAIL needs every term to be a clear non-match."""
    if PASS in results:
        return PASS
    return UNKNOWN if UNKNOWN in results else FAIL


def _combine_exclusion(results: list[CriterionResultEnum]) -> CriterionResultEnum:
    """Results are per-term 'patient is in this excluded group' (PASS = in it)."""
    if PASS in results:
        return FAIL
    return UNKNOWN if UNKNOWN in results else PASS


def _combine_sides(included: list[CriterionResultEnum], excluded: list[CriterionResultEnum]) -> CriterionResultEnum:
    """Merge inclusion and exclusion outcomes; FAIL dominates, then UNKNOWN."""
    sides = []
    if included:
        sides.append(_combine_inclusion(included))
    if excluded:
        sides.append(_combine_exclusion(excluded))
    if FAIL in sides:
        return FAIL
    return UNKNOWN if UNKNOWN in sides else PASS


def _eval_age(patient: PatientProfile, criteria: TrialCriteria) -> CriterionEvaluation | None:
    if criteria.min_age_years is None and criteria.max_age_years is None:
        return None
    parts = []
    if criteria.min_age_years is not None:
        parts.append(f">= {criteria.min_age_years}")
    if criteria.max_age_years is not None:
        parts.append(f"<= {criteria.max_age_years}")
    desc = "Age " + " and ".join(parts)

    fail = (criteria.min_age_years is not None and patient.age_years < criteria.min_age_years) or (
        criteria.max_age_years is not None and patient.age_years > criteria.max_age_years
    )
    return CriterionEvaluation(
        field_name="age",
        requirement_description=desc,
        patient_value=str(patient.age_years),
        result=FAIL if fail else PASS,
    )


def _eval_sex(patient: PatientProfile, criteria: TrialCriteria) -> CriterionEvaluation | None:
    if criteria.sex is None or criteria.sex == Sex.ALL:
        return None
    desc = f"Sex: {criteria.sex.value}"
    if patient.sex == Sex.ALL:
        result = UNKNOWN
    else:
        result = PASS if patient.sex == criteria.sex else FAIL
    return CriterionEvaluation(
        field_name="sex", requirement_description=desc, patient_value=patient.sex.value, result=result
    )


def _eval_ecog(patient: PatientProfile, criteria: TrialCriteria) -> CriterionEvaluation | None:
    if criteria.ecog_max is None:
        return None
    desc = f"ECOG performance status <= {criteria.ecog_max}"
    if patient.ecog is None:
        return CriterionEvaluation(field_name="ecog", requirement_description=desc, patient_value=None, result=UNKNOWN)
    result = PASS if patient.ecog <= criteria.ecog_max else FAIL
    return CriterionEvaluation(
        field_name="ecog", requirement_description=desc, patient_value=str(patient.ecog), result=result
    )


def _eval_histology(patient: PatientProfile, criteria: TrialCriteria) -> CriterionEvaluation | None:
    if not criteria.histology_included and not criteria.histology_excluded:
        return None
    parts = []
    if criteria.histology_included:
        parts.append(f"histology in {criteria.histology_included}")
    if criteria.histology_excluded:
        parts.append(f"histology not in {criteria.histology_excluded}")
    desc = "; ".join(parts)

    if not patient.histology or _norm(patient.histology) in UNINFORMATIVE_PATIENT_VALUES:
        return CriterionEvaluation(
            field_name="histology", requirement_description=desc, patient_value=patient.histology, result=UNKNOWN
        )

    patient_norm = _norm_histology(patient.histology)

    def relation(term: str, excluding: bool = False) -> CriterionResultEnum:
        canonical = _norm_histology(term)
        # An exclusion can't plausibly mean all of NSCLC in an NSCLC trial.
        if not _is_known_histology(canonical) or (excluding and canonical == "nsclc"):
            return UNKNOWN
        return PASS if _histology_matches(patient_norm, canonical) else FAIL

    return CriterionEvaluation(
        field_name="histology",
        requirement_description=desc,
        patient_value=patient.histology,
        result=_combine_sides(
            [relation(t) for t in criteria.histology_included],
            [relation(t, excluding=True) for t in criteria.histology_excluded],
        ),
    )


def _eval_stage(patient: PatientProfile, criteria: TrialCriteria) -> CriterionEvaluation | None:
    if not criteria.stage_included and not criteria.stage_excluded:
        return None
    parts = []
    if criteria.stage_included:
        parts.append(f"stage in {criteria.stage_included}")
    if criteria.stage_excluded:
        parts.append(f"stage not in {criteria.stage_excluded}")
    desc = "; ".join(parts)

    patient_stage = _parse_stage(patient.stage) if patient.stage else None
    if patient_stage is None:
        return CriterionEvaluation(
            field_name="stage", requirement_description=desc, patient_value=patient.stage, result=UNKNOWN
        )

    def relation(term: str) -> CriterionResultEnum:
        parsed = _parse_stage(term)
        return UNKNOWN if parsed is None else _stage_relation(patient_stage, parsed)

    return CriterionEvaluation(
        field_name="stage",
        requirement_description=desc,
        patient_value=patient.stage,
        result=_combine_sides(
            [relation(t) for t in criteria.stage_included],
            [relation(t) for t in criteria.stage_excluded],
        ),
    )


def _same_variant(required: str, patient_detail: str | None) -> bool:
    if not patient_detail:
        return False
    a, b = _norm(required), _norm(patient_detail)
    return a in b or b in a


def _eval_biomarkers(patient: PatientProfile, criteria: TrialCriteria) -> list[CriterionEvaluation]:
    evaluations = []
    patient_by_gene = {b.gene: b for b in patient.biomarkers}

    for req in criteria.biomarkers:
        patient_bm = patient_by_gene.get(req.gene)
        detail_suffix = f" ({req.detail})" if req.detail else ""
        desc = f"{req.gene.value} {req.status.value}{detail_suffix}"
        field_name = f"biomarker_{req.gene.value}"

        if patient_bm is None or patient_bm.status == PatientBiomarkerStatus.UNKNOWN:
            evaluations.append(
                CriterionEvaluation(
                    field_name=field_name,
                    requirement_description=desc,
                    patient_value=patient_bm.status.value if patient_bm else None,
                    result=UNKNOWN,
                )
            )
            continue

        patient_positive = patient_bm.status == PatientBiomarkerStatus.POSITIVE
        if patient_positive and req.detail and not _same_variant(req.detail, patient_bm.detail):
            # The requirement names a variant; a positive gene result doesn't say
            # whether this patient has that one.
            evaluations.append(
                CriterionEvaluation(
                    field_name=field_name,
                    requirement_description=desc,
                    patient_value=patient_bm.detail or patient_bm.status.value,
                    result=UNKNOWN,
                )
            )
            continue

        if req.status == RequirementStatus.REQUIRED_POSITIVE:
            result = PASS if patient_positive else FAIL
        elif req.status == RequirementStatus.REQUIRED_NEGATIVE:
            result = PASS if not patient_positive else FAIL
        else:  # EXCLUDED_IF_POSITIVE
            result = FAIL if patient_positive else PASS

        evaluations.append(
            CriterionEvaluation(
                field_name=field_name,
                requirement_description=desc,
                patient_value=patient_bm.status.value,
                result=result,
            )
        )

    return evaluations


def _eval_pdl1(patient: PatientProfile, criteria: TrialCriteria) -> CriterionEvaluation | None:
    if criteria.pdl1_tps_min is None and criteria.pdl1_tps_max is None:
        return None
    parts = []
    if criteria.pdl1_tps_min is not None:
        parts.append(f">= {criteria.pdl1_tps_min}%")
    if criteria.pdl1_tps_max is not None:
        parts.append(f"<= {criteria.pdl1_tps_max}%")
    desc = "PD-L1 TPS " + " and ".join(parts)

    if patient.pdl1_tps is None:
        return CriterionEvaluation(
            field_name="pdl1_tps", requirement_description=desc, patient_value=None, result=UNKNOWN
        )

    fail = (criteria.pdl1_tps_min is not None and patient.pdl1_tps < criteria.pdl1_tps_min) or (
        criteria.pdl1_tps_max is not None and patient.pdl1_tps > criteria.pdl1_tps_max
    )
    return CriterionEvaluation(
        field_name="pdl1_tps",
        requirement_description=desc,
        patient_value=str(patient.pdl1_tps),
        result=FAIL if fail else PASS,
    )


def _eval_prior_lines(patient: PatientProfile, criteria: TrialCriteria) -> CriterionEvaluation | None:
    if criteria.prior_therapy_lines_min is None and criteria.prior_therapy_lines_max is None:
        return None
    parts = []
    if criteria.prior_therapy_lines_min is not None:
        parts.append(f">= {criteria.prior_therapy_lines_min}")
    if criteria.prior_therapy_lines_max is not None:
        parts.append(f"<= {criteria.prior_therapy_lines_max}")
    desc = "Prior therapy lines " + " and ".join(parts)

    if not patient.prior_treatment_history_provided or patient.prior_therapy_lines is None:
        return CriterionEvaluation(
            field_name="prior_therapy_lines", requirement_description=desc, patient_value=None, result=UNKNOWN
        )

    fail = (
        criteria.prior_therapy_lines_min is not None
        and patient.prior_therapy_lines < criteria.prior_therapy_lines_min
    ) or (
        criteria.prior_therapy_lines_max is not None
        and patient.prior_therapy_lines > criteria.prior_therapy_lines_max
    )
    return CriterionEvaluation(
        field_name="prior_therapy_lines",
        requirement_description=desc,
        patient_value=str(patient.prior_therapy_lines),
        result=FAIL if fail else PASS,
    )


def _eval_treatment_naive(patient: PatientProfile, criteria: TrialCriteria) -> CriterionEvaluation | None:
    if criteria.treatment_naive_required is None:
        return None
    desc = (
        "Treatment-naive required"
        if criteria.treatment_naive_required
        else "Prior systemic treatment required (not treatment-naive)"
    )

    if not patient.prior_treatment_history_provided:
        return CriterionEvaluation(
            field_name="treatment_naive", requirement_description=desc, patient_value=None, result=UNKNOWN
        )

    patient_is_naive = not patient.prior_treatments_received and patient.prior_therapy_lines in (0, None)
    if criteria.treatment_naive_required:
        result = PASS if patient_is_naive else FAIL
    else:
        result = PASS if not patient_is_naive else FAIL

    return CriterionEvaluation(
        field_name="treatment_naive",
        requirement_description=desc,
        patient_value="treatment-naive" if patient_is_naive else "has prior therapy",
        result=result,
    )


def _eval_prior_treatment_lists(patient: PatientProfile, criteria: TrialCriteria) -> list[CriterionEvaluation]:
    evaluations = []
    patient_treatments_norm = {_norm(t) for t in patient.prior_treatments_received}
    patient_value_str = ", ".join(patient.prior_treatments_received) or None

    def _matches(drug_norm: str) -> bool:
        return any(drug_norm in t or t in drug_norm for t in patient_treatments_norm)

    for drug in criteria.required_prior_treatments:
        desc = f"Must have received prior {drug}"
        field_name = f"required_prior_treatment_{_norm(drug)}"
        if not patient.prior_treatment_history_provided:
            evaluations.append(
                CriterionEvaluation(field_name=field_name, requirement_description=desc, patient_value=None, result=UNKNOWN)
            )
            continue
        if _matches(_norm(drug)):
            result = PASS
        else:
            # An empty history is a real contradiction. A non-empty one that doesn't
            # contain the string isn't: "carboplatin" satisfies "platinum-based
            # chemotherapy", and these lists have no drug-class knowledge.
            result = UNKNOWN if patient_treatments_norm else FAIL
        evaluations.append(
            CriterionEvaluation(
                field_name=field_name, requirement_description=desc, patient_value=patient_value_str, result=result
            )
        )

    for drug in criteria.excluded_prior_treatments:
        desc = f"Must NOT have received prior {drug}"
        field_name = f"excluded_prior_treatment_{_norm(drug)}"
        if not patient.prior_treatment_history_provided:
            evaluations.append(
                CriterionEvaluation(field_name=field_name, requirement_description=desc, patient_value=None, result=UNKNOWN)
            )
            continue
        if _matches(_norm(drug)):
            result = FAIL
        else:
            # Same reasoning: "sotorasib" doesn't string-match "any KRAS G12C inhibitor",
            # so no match can't be reported as a pass.
            result = UNKNOWN if patient_treatments_norm else PASS
        evaluations.append(
            CriterionEvaluation(
                field_name=field_name, requirement_description=desc, patient_value=patient_value_str, result=result
            )
        )

    return evaluations


def _eval_brain_mets(patient: PatientProfile, criteria: TrialCriteria) -> CriterionEvaluation | None:
    if criteria.brain_mets_status in (None, BrainMetsStatus.UNKNOWN):
        return None
    desc = f"Brain metastases: trial allows up to '{criteria.brain_mets_status.value}'"

    if patient.brain_mets_status == BrainMetsStatus.UNKNOWN:
        return CriterionEvaluation(
            field_name="brain_mets", requirement_description=desc, patient_value=None, result=UNKNOWN
        )

    allowed_level = BRAIN_METS_ORDER[criteria.brain_mets_status]
    patient_level = BRAIN_METS_ORDER[patient.brain_mets_status]
    result = PASS if patient_level <= allowed_level else FAIL
    return CriterionEvaluation(
        field_name="brain_mets",
        requirement_description=desc,
        patient_value=patient.brain_mets_status.value,
        result=result,
    )


def _eval_measurable_disease(patient: PatientProfile, criteria: TrialCriteria) -> CriterionEvaluation | None:
    if not criteria.measurable_disease_required:
        return None
    desc = "Measurable disease required"

    if patient.measurable_disease is None:
        return CriterionEvaluation(
            field_name="measurable_disease", requirement_description=desc, patient_value=None, result=UNKNOWN
        )

    result = PASS if patient.measurable_disease else FAIL
    return CriterionEvaluation(
        field_name="measurable_disease",
        requirement_description=desc,
        patient_value=str(patient.measurable_disease),
        result=result,
    )


def _eval_labs(patient: PatientProfile, criteria: TrialCriteria) -> list[CriterionEvaluation]:
    evaluations = []
    patient_labs_norm = {_norm_lab(k): v for k, v in patient.organ_function_labs.items()}

    for lab in criteria.organ_function_labs:
        key = _norm_lab(lab.lab_name)
        unit_suffix = f" {lab.unit}" if lab.unit else ""
        desc = f"{lab.lab_name} {lab.operator} {lab.value}{unit_suffix}"
        field_name = f"lab_{key}"

        patient_val = patient_labs_norm.get(key)
        if patient_val is None:
            evaluations.append(
                CriterionEvaluation(field_name=field_name, requirement_description=desc, patient_value=None, result=UNKNOWN)
            )
            continue

        op_fn = LAB_OPERATORS.get(lab.operator)
        # ULN-relative and other-unit thresholds can't be compared to an absolute value.
        if op_fn is None or not _lab_unit_comparable(key, lab.unit):
            evaluations.append(
                CriterionEvaluation(
                    field_name=field_name, requirement_description=desc, patient_value=str(patient_val), result=UNKNOWN
                )
            )
            continue

        result = PASS if op_fn(patient_val, lab.value) else FAIL
        evaluations.append(
            CriterionEvaluation(
                field_name=field_name, requirement_description=desc, patient_value=str(patient_val), result=result
            )
        )

    return evaluations


_SINGLE_EVALUATORS = [
    _eval_age,
    _eval_sex,
    _eval_ecog,
    _eval_histology,
    _eval_stage,
    _eval_pdl1,
    _eval_prior_lines,
    _eval_treatment_naive,
    _eval_brain_mets,
    _eval_measurable_disease,
]


def _build_explanation(
    passed: list[CriterionEvaluation],
    failed: list[CriterionEvaluation],
    unknown: list[CriterionEvaluation],
    manual_review_items: list[str],
) -> str:
    parts = [f"{len(passed)} criteria passed, {len(failed)} failed, {len(unknown)} could not be evaluated."]
    if failed:
        parts.append(
            "Failed: " + "; ".join(f"{e.field_name} ({e.requirement_description})" for e in failed) + "."
        )
    if unknown:
        parts.append("Needs patient data for: " + "; ".join(e.field_name for e in unknown) + ".")
    if manual_review_items:
        parts.append(f"{len(manual_review_items)} eligibility statement(s) require manual review.")
    return " ".join(parts)


def match_patient_to_trial(patient: PatientProfile, criteria: TrialCriteria) -> MatchResult:
    """Deterministically evaluate a patient against one trial's structured criteria."""

    evaluations: list[CriterionEvaluation] = []
    for evaluator in _SINGLE_EVALUATORS:
        result = evaluator(patient, criteria)
        if result is not None:
            evaluations.append(result)

    evaluations.extend(_eval_biomarkers(patient, criteria))
    evaluations.extend(_eval_prior_treatment_lists(patient, criteria))
    evaluations.extend(_eval_labs(patient, criteria))

    passed = [e for e in evaluations if e.result == PASS]
    failed = [e for e in evaluations if e.result == FAIL]
    unknown = [e for e in evaluations if e.result == UNKNOWN]

    manual_review_items = list(criteria.unstructured_inclusion) + list(criteria.unstructured_exclusion)

    if failed:
        status = MatchStatus.NOT_ELIGIBLE
    elif criteria.manual_review_flag or unknown:
        status = MatchStatus.NEEDS_REVIEW
    else:
        status = MatchStatus.LIKELY_ELIGIBLE

    return MatchResult(
        nct_id=criteria.nct_id,
        trial_title=criteria.trial_title,
        overall_status=criteria.overall_status,
        match_status=status,
        passed_criteria=passed,
        failed_criteria=failed,
        unknown_criteria=unknown,
        manual_review_items=manual_review_items,
        explanation=_build_explanation(passed, failed, unknown, manual_review_items),
        ctgov_url=f"https://clinicaltrials.gov/study/{criteria.nct_id}",
    )
