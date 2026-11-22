"""Field-level scoring of a predicted TrialCriteria against a hand-labeled gold one.

The LLM parse (free text -> TrialCriteria) is the only probabilistic step in the
system, so it is the thing this harness grades. Matching (patient x criteria) is
deterministic and is covered by tests/, not here.

Design: TrialCriteria has four structurally different kinds of field, and they fail
in different ways, so each is scored differently rather than with one blunt equality
check:

  1. SCALARS  (min_age_years, ecog_max, pdl1_tps_min, ...)
       Exact match. But None-vs-value is called out as its own error class:
       - missed:      gold has a value, prediction is None  (criterion dropped)
       - hallucinated: gold is None, prediction has a value (criterion invented)
       - wrong_value: both present but unequal
     These three are very different failure modes and are tracked separately.

  2. SETS  (histology_included, stage_excluded, ...)
       Order-insensitive precision/recall AFTER normalization, reusing matcher.py's
       own _norm_stage / _norm_histology so "Stage IV" vs "metastatic" is not a
       false miss. Prior-treatment lists hold free-text clauses, so they use the same
       fuzzy item coverage as the unstructured buckets.

  3. KEYED OBJECT LISTS  (biomarkers keyed by gene, organ_function_labs by lab_name)
       Align gold and predicted items by key, then compare sub-fields. A trial can
       state several requirements for one gene, so a gene's statuses compare as a set.
       A biomarker whose requirement-status is flipped (REQUIRED_POSITIVE vs
       EXCLUDED_IF_POSITIVE) is marked high-severity, because that single field flips
       eligibility. The free-text `detail` is not scored.

  4. UNSTRUCTURED BUCKETS  (unstructured_inclusion/exclusion)
       The model's judgment about what could NOT be structured. Scored in both
       directions: over-structuring (gold flagged it for review, prediction confidently
       structured it away -> dangerous) and over-flagging (gold structured it,
       prediction punted to review -> tool becomes useless). Bucket items are "near
       verbatim" by design, so they align by word overlap, not string equality.

Every per-field result carries a severity so the report can rank failures by how much
they'd distort a real eligibility decision.

Most schema fields are empty for any given trial, and null-vs-null scores as correct,
so overall accuracy is dominated by fields neither side mentions. Each score therefore
records whether it is `populated` (gold or prediction has a value), and accuracy over
populated fields is the number to read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# The matcher's own normalizers, so scoring agrees with how matching treats these strings.
from matcher import _is_known_histology, _norm, _norm_histology, _norm_lab, _norm_stage
from schema import TrialCriteria


class Outcome(str, Enum):
    CORRECT = "correct"
    MISSED = "missed"            # gold had it, prediction dropped it
    HALLUCINATED = "hallucinated"  # prediction invented it, gold had none
    WRONG_VALUE = "wrong_value"    # both present, disagree


class Severity(str, Enum):
    # eligibility-flipping errors (wrong biomarker direction, missed exclusion)
    CRITICAL = "critical"
    # narrows/widens the pool but won't silently flip a clear decision
    MODERATE = "moderate"
    # cosmetic / low impact (e.g. a missed lab that matching treats as UNKNOWN anyway)
    MINOR = "minor"


class FieldType(str, Enum):
    SCALAR = "scalar"
    SET = "set"
    OBJECT_LIST = "object_list"
    UNSTRUCTURED = "unstructured"


@dataclass
class FieldScore:
    field_name: str
    field_type: FieldType
    outcome: Outcome
    severity: Severity
    gold: Any = None
    predicted: Any = None
    note: str = ""
    populated: bool = True  # False when gold and prediction are both empty
    n_missing: int = 0  # list fields: gold items with no counterpart in the prediction
    n_extra: int = 0  # list fields: predicted items with no counterpart in gold

    @property
    def correct(self) -> bool:
        return self.outcome == Outcome.CORRECT


@dataclass
class TrialScore:
    nct_id: str
    field_scores: list[FieldScore] = field(default_factory=list)

    @property
    def n_fields(self) -> int:
        return len(self.field_scores)

    @property
    def n_correct(self) -> int:
        return sum(1 for f in self.field_scores if f.correct)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_fields if self.n_fields else 1.0

    def unstructured_items(self) -> tuple[int, int, int, int]:
        """(gold items, gold items matched, predicted items, predicted items matched)."""
        g = m = p = x = 0
        for f in self.field_scores:
            if f.field_type == FieldType.UNSTRUCTURED:
                g += f.gold
                m += f.gold - f.n_missing
                p += f.predicted
                x += f.predicted - f.n_extra
        return g, m, p, x

    @property
    def populated_accuracy(self) -> float:
        populated = [f for f in self.field_scores if f.populated]
        return sum(1 for f in populated if f.correct) / len(populated) if populated else 1.0

    def by_outcome(self, outcome: Outcome) -> list[FieldScore]:
        return [f for f in self.field_scores if f.outcome == outcome]

    def critical_errors(self) -> list[FieldScore]:
        return [f for f in self.field_scores if not f.correct and f.severity == Severity.CRITICAL]


# --- scalar scoring ---------------------------------------------------------

# Scalars whose absence/mismatch meaningfully changes the eligibility pool.
_SCALAR_SEVERITY = {
    "min_age_years": Severity.MODERATE,
    "max_age_years": Severity.MODERATE,
    "ecog_max": Severity.MODERATE,
    "pdl1_tps_min": Severity.MODERATE,
    "pdl1_tps_max": Severity.MODERATE,
    "prior_therapy_lines_min": Severity.MODERATE,
    "prior_therapy_lines_max": Severity.MODERATE,
    "treatment_naive_required": Severity.CRITICAL,   # flips whole populations
    "measurable_disease_required": Severity.MINOR,
    "brain_mets_status": Severity.CRITICAL,          # exclusion-flipping
    "sex": Severity.MODERATE,
}

_SCALAR_FIELDS = list(_SCALAR_SEVERITY.keys())


def _score_scalar(name: str, gold: Any, pred: Any) -> FieldScore:
    sev = _SCALAR_SEVERITY.get(name, Severity.MODERATE)
    # normalize enums to their value for comparison
    g = gold.value if isinstance(gold, Enum) else gold
    p = pred.value if isinstance(pred, Enum) else pred
    if name == "sex":  # an unstated sex and "all" both mean no restriction
        g = None if g == "all" else g
        p = None if p == "all" else p

    if g == p:
        outcome = Outcome.CORRECT
    elif g is not None and p is None:
        outcome = Outcome.MISSED
    elif g is None and p is not None:
        outcome = Outcome.HALLUCINATED
    else:
        outcome = Outcome.WRONG_VALUE

    # A correct field is never elevated in severity.
    if outcome == Outcome.CORRECT:
        sev = Severity.MINOR
    return FieldScore(name, FieldType.SCALAR, outcome, sev, gold=g, predicted=p,
                      populated=g is not None or p is not None)


# --- set scoring ------------------------------------------------------------

_SET_NORMALIZERS = {
    "histology_included": _norm_histology,
    "histology_excluded": _norm_histology,
    "stage_included": _norm_stage,
    "stage_excluded": _norm_stage,
    "required_prior_treatments": _norm,
    "excluded_prior_treatments": _norm,
}

# Excluded-list misses are more dangerous than included-list misses: dropping an
# exclusion can turn a true "Not Eligible" into a false "Likely Eligible".
_SET_SEVERITY = {
    "histology_included": Severity.MODERATE,
    "histology_excluded": Severity.CRITICAL,
    "stage_included": Severity.MODERATE,
    "stage_excluded": Severity.CRITICAL,
    "required_prior_treatments": Severity.MODERATE,
    "excluded_prior_treatments": Severity.CRITICAL,
}


_FUZZY_SETS = {"required_prior_treatments", "excluded_prior_treatments"}
_STAGE_CANONICAL = re.compile(r"^(iv|iii|ii|i)[abc]?$")
_CANONICAL = {
    "stage_included": lambda x: bool(_STAGE_CANONICAL.match(x)),
    "stage_excluded": lambda x: bool(_STAGE_CANONICAL.match(x)),
    "histology_included": _is_known_histology,
    "histology_excluded": _is_known_histology,
}


def _score_set(name: str, gold: list[str], pred: list[str]) -> FieldScore:
    norm = _SET_NORMALIZERS.get(name, _norm)
    g = {norm(x) for x in (gold or [])}
    p = {norm(x) for x in (pred or [])}
    populated = bool(g or p)

    if name in _FUZZY_SETS:
        missing, extra = _unaligned(g, p)
    else:
        # Terms that normalize to a known stage or histology compare exactly (so "squamous"
        # never matches "non_squamous"); anything left as free text aligns by wording.
        is_canonical = _CANONICAL.get(name, lambda _: True)
        g_canon, p_canon = {x for x in g if is_canonical(x)}, {x for x in p if is_canonical(x)}
        free_missing, free_extra = _unaligned(g - g_canon, p - p_canon)
        missing = sorted(g_canon - p_canon) + free_missing
        extra = sorted(p_canon - g_canon) + free_extra

    if not missing and not extra:
        return FieldScore(name, FieldType.SET, Outcome.CORRECT, Severity.MINOR, gold=sorted(g), predicted=sorted(p),
                          populated=populated)

    if missing and not extra:
        outcome = Outcome.MISSED
    elif extra and not missing:
        outcome = Outcome.HALLUCINATED
    else:
        outcome = Outcome.WRONG_VALUE

    sev = _SET_SEVERITY.get(name, Severity.MODERATE)
    note = f"missing={sorted(missing)} extra={sorted(extra)}"
    return FieldScore(name, FieldType.SET, outcome, sev, gold=sorted(g), predicted=sorted(p), note=note,
                      populated=populated, n_missing=len(missing), n_extra=len(extra))


# --- keyed object-list scoring ---------------------------------------------


def _statuses_by_gene(items: list) -> dict:
    by_gene: dict = {}
    for item in items:
        by_gene.setdefault(item.gene, set()).add(item.status.value)
    return by_gene


def _score_biomarkers(gold: list, pred: list) -> list[FieldScore]:
    scores: list[FieldScore] = []
    g_by_gene = _statuses_by_gene(gold)
    p_by_gene = _statuses_by_gene(pred)

    for gene in sorted(set(g_by_gene) | set(p_by_gene), key=lambda g: g.value):
        fname = f"biomarker_{gene.value}"
        gs = g_by_gene.get(gene)
        ps = p_by_gene.get(gene)
        g_str = "+".join(sorted(gs)) if gs else None
        p_str = "+".join(sorted(ps)) if ps else None

        if gs and not ps:
            scores.append(FieldScore(fname, FieldType.OBJECT_LIST, Outcome.MISSED, Severity.CRITICAL,
                                     gold=g_str, predicted=None,
                                     note="biomarker requirement dropped"))
        elif ps and not gs:
            scores.append(FieldScore(fname, FieldType.OBJECT_LIST, Outcome.HALLUCINATED, Severity.CRITICAL,
                                     gold=None, predicted=p_str,
                                     note="biomarker requirement invented"))
        elif gs != ps:
            # the eligibility-flipping error class
            scores.append(FieldScore(fname, FieldType.OBJECT_LIST, Outcome.WRONG_VALUE, Severity.CRITICAL,
                                     gold=g_str, predicted=p_str,
                                     note="requirement direction flipped"))
        else:
            scores.append(FieldScore(fname, FieldType.OBJECT_LIST, Outcome.CORRECT, Severity.MINOR,
                                     gold=g_str, predicted=p_str))
    return scores


def _score_labs(gold: list, pred: list) -> list[FieldScore]:
    scores: list[FieldScore] = []
    g_by = {_norm_lab(l.lab_name): l for l in gold}
    p_by = {_norm_lab(l.lab_name): l for l in pred}
    all_labs = sorted(set(g_by) | set(p_by))

    for key in all_labs:
        fname = f"lab_{key}"
        gl = g_by.get(key)
        pl = p_by.get(key)
        # Missed labs are MINOR: matcher treats an absent lab as UNKNOWN, not a false pass.
        if gl and not pl:
            scores.append(FieldScore(fname, FieldType.OBJECT_LIST, Outcome.MISSED, Severity.MINOR,
                                     gold=f"{gl.operator}{gl.value}", predicted=None))
        elif pl and not gl:
            scores.append(FieldScore(fname, FieldType.OBJECT_LIST, Outcome.HALLUCINATED, Severity.MODERATE,
                                     gold=None, predicted=f"{pl.operator}{pl.value}"))
        elif (gl.operator, gl.value) != (pl.operator, pl.value):
            scores.append(FieldScore(fname, FieldType.OBJECT_LIST, Outcome.WRONG_VALUE, Severity.MODERATE,
                                     gold=f"{gl.operator}{gl.value}", predicted=f"{pl.operator}{pl.value}"))
        else:
            scores.append(FieldScore(fname, FieldType.OBJECT_LIST, Outcome.CORRECT, Severity.MINOR,
                                     gold=f"{gl.operator}{gl.value}", predicted=f"{pl.operator}{pl.value}"))
    return scores


# --- unstructured-bucket scoring -------------------------------------------


_STOPWORDS = {
    "a", "an", "and", "any", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "must",
    "no", "not", "of", "on", "or", "other", "the", "to", "with", "within", "who", "patients", "participants",
    "after", "before", "days", "dose", "first", "history", "months", "past", "prior", "week", "weeks",
}
_ALIGN_THRESHOLD = 0.66


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and not t.isdigit()}


def _similarity(a: str, b: str) -> float:
    """Word overlap. Uses the shorter item as denominator so that one prediction
    merging two gold criteria (or splitting one) still aligns."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return float(a == b)
    return len(ta & tb) / min(len(ta), len(tb))


def _unaligned(gold: set[str], pred: set[str]) -> tuple[list[str], list[str]]:
    """Items on each side with no counterpart on the other: (missing, extra)."""
    missing = [g for g in sorted(gold) if not any(_similarity(g, p) >= _ALIGN_THRESHOLD for p in pred)]
    extra = [p for p in sorted(pred) if not any(_similarity(g, p) >= _ALIGN_THRESHOLD for g in gold)]
    return missing, extra


def _score_unstructured(name: str, gold: list[str], pred: list[str]) -> FieldScore:
    g, p = {_norm(x) for x in (gold or [])}, {_norm(x) for x in (pred or [])}
    missing, extra = _unaligned(g, p)
    common = dict(gold=len(g), predicted=len(p), populated=bool(g or p), n_missing=len(missing), n_extra=len(extra))

    if not missing and not extra:
        return FieldScore(name, FieldType.UNSTRUCTURED, Outcome.CORRECT, Severity.MINOR, **common)
    if missing and not extra:
        # over-structuring: gold flagged text for review, prediction didn't
        return FieldScore(name, FieldType.UNSTRUCTURED, Outcome.MISSED, Severity.MODERATE,
                          note=f"over-structured: {len(missing)} review item(s) not flagged", **common)
    if extra and not missing:
        # over-flagging: prediction punted things to review that gold structured
        return FieldScore(name, FieldType.UNSTRUCTURED, Outcome.HALLUCINATED, Severity.MINOR,
                          note=f"over-flagged: {len(extra)} extra review item(s)", **common)
    return FieldScore(name, FieldType.UNSTRUCTURED, Outcome.WRONG_VALUE, Severity.MODERATE,
                      note=f"{len(missing)} review item(s) not flagged, {len(extra)} extra", **common)


# --- top level --------------------------------------------------------------


def score_trial(gold: TrialCriteria, pred: TrialCriteria) -> TrialScore:
    """Score one predicted TrialCriteria against its gold label, field by field."""
    scores: list[FieldScore] = []

    for name in _SCALAR_FIELDS:
        scores.append(_score_scalar(name, getattr(gold, name), getattr(pred, name)))

    for name in _SET_NORMALIZERS:
        scores.append(_score_set(name, getattr(gold, name), getattr(pred, name)))

    scores.extend(_score_biomarkers(gold.biomarkers, pred.biomarkers))
    scores.extend(_score_labs(gold.organ_function_labs, pred.organ_function_labs))

    for name in ("unstructured_inclusion", "unstructured_exclusion"):
        scores.append(_score_unstructured(name, getattr(gold, name), getattr(pred, name)))

    return TrialScore(nct_id=gold.nct_id, field_scores=scores)
