"""Pydantic schemas shared across the pipeline: LLM extraction target, cached trial
criteria, patient profile, and match results."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# Bump whenever TrialCriteriaExtracted's shape changes -- cache.py treats a stored
# row with an older version as a miss so schema iterations don't serve stale data.
SCHEMA_VERSION = 2


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    ALL = "all"


class BiomarkerGene(str, Enum):
    EGFR = "EGFR"
    ALK = "ALK"
    ROS1 = "ROS1"
    KRAS_G12C = "KRAS_G12C"
    BRAF = "BRAF"
    MET = "MET"
    RET = "RET"
    NTRK = "NTRK"
    HER2 = "HER2"


class RequirementStatus(str, Enum):
    REQUIRED_POSITIVE = "required_positive"
    REQUIRED_NEGATIVE = "required_negative"
    EXCLUDED_IF_POSITIVE = "excluded_if_positive"


class PatientBiomarkerStatus(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class BrainMetsStatus(str, Enum):
    NONE = "none"
    TREATED_STABLE = "treated_stable"
    UNTREATED = "untreated"
    SYMPTOMATIC = "symptomatic"
    UNKNOWN = "unknown"


class BiomarkerRequirement(BaseModel):
    gene: BiomarkerGene
    status: RequirementStatus
    detail: Optional[str] = Field(
        None, description="e.g. 'exon 19 deletion or L858R', verbatim if the trial names a specific variant"
    )


class LabRequirement(BaseModel):
    lab_name: str = Field(
        description="e.g. ANC, platelets, hemoglobin, creatinine_clearance, bilirubin, AST, ALT"
    )
    operator: str = Field(description="One of: >=, <=, >, <")
    value: float
    unit: Optional[str] = None


# NOTE: Claude's structured-outputs endpoint (messages.parse(output_format=...)) has an
# undocumented limit on the number of *top-level* properties in the schema -- a flat
# model with ~13+ top-level optional fields is rejected with "Schema is too complex.",
# confirmed empirically against the live API. Grouping related fields into nested
# sub-models keeps the top-level property count low (8 here) while the total field
# count is unchanged, which the API accepts. TrialCriteriaExtracted is therefore
# grouped for the LLM; TrialCriteria (below) stays flat for matcher.py to consume --
# parse_criteria.py does the flattening.


class DemographicsExtracted(BaseModel):
    min_age_years: Optional[float] = None
    max_age_years: Optional[float] = None
    sex: Optional[Sex] = None
    ecog_max: Optional[int] = Field(None, description="Highest ECOG PS allowed, e.g. 1 means 0-1 allowed")


class DiseaseExtracted(BaseModel):
    histology_included: list[str] = Field(default_factory=list)
    histology_excluded: list[str] = Field(default_factory=list)
    stage_included: list[str] = Field(default_factory=list, description="e.g. ['IIIB','IV','metastatic']")
    stage_excluded: list[str] = Field(default_factory=list)
    pdl1_tps_min: Optional[float] = None
    pdl1_tps_max: Optional[float] = None
    measurable_disease_required: Optional[bool] = None
    brain_mets_status: Optional[BrainMetsStatus] = Field(
        None, description="The trial's stance on brain mets, or null if not addressed"
    )


class TherapyHistoryExtracted(BaseModel):
    prior_therapy_lines_min: Optional[int] = None
    prior_therapy_lines_max: Optional[int] = None
    treatment_naive_required: Optional[bool] = None
    required_prior_treatments: list[str] = Field(default_factory=list)
    excluded_prior_treatments: list[str] = Field(default_factory=list)


class TrialCriteriaExtracted(BaseModel):
    """The schema Claude fills in from a trial's free-text eligibility criteria.

    Only extract what is explicitly stated -- never infer or guess a value. Anything
    that doesn't map cleanly onto a field below goes into the unstructured_* lists
    instead of being forced or dropped.
    """

    demographics: DemographicsExtracted = Field(default_factory=DemographicsExtracted)
    disease: DiseaseExtracted = Field(default_factory=DiseaseExtracted)
    therapy_history: TherapyHistoryExtracted = Field(default_factory=TherapyHistoryExtracted)
    biomarkers: list[BiomarkerRequirement] = Field(default_factory=list)
    organ_function_labs: list[LabRequirement] = Field(default_factory=list)
    unstructured_inclusion: list[str] = Field(
        default_factory=list,
        description="Inclusion criteria that don't map to the fields above -- copy near-verbatim",
    )
    unstructured_exclusion: list[str] = Field(
        default_factory=list,
        description="Exclusion criteria that don't map to the fields above -- copy near-verbatim",
    )
    parse_confidence: Optional[str] = Field(
        None, description="'high' | 'medium' | 'low' -- your confidence this extraction is complete and correct"
    )


class TrialCriteria(BaseModel):
    """Persisted/matched version of TrialCriteriaExtracted: the same fields, flattened
    to one level (see the note above TrialCriteriaExtracted for why the LLM-facing
    schema is grouped but this one isn't), plus trial identity. parse_criteria.py
    builds this from a TrialCriteriaExtracted by unpacking the nested groups."""

    nct_id: str
    trial_title: str = ""
    overall_status: str = ""

    min_age_years: Optional[float] = None
    max_age_years: Optional[float] = None
    sex: Optional[Sex] = None
    ecog_max: Optional[int] = None
    histology_included: list[str] = Field(default_factory=list)
    histology_excluded: list[str] = Field(default_factory=list)
    stage_included: list[str] = Field(default_factory=list)
    stage_excluded: list[str] = Field(default_factory=list)
    biomarkers: list[BiomarkerRequirement] = Field(default_factory=list)
    pdl1_tps_min: Optional[float] = None
    pdl1_tps_max: Optional[float] = None
    prior_therapy_lines_min: Optional[int] = None
    prior_therapy_lines_max: Optional[int] = None
    treatment_naive_required: Optional[bool] = None
    required_prior_treatments: list[str] = Field(default_factory=list)
    excluded_prior_treatments: list[str] = Field(default_factory=list)
    brain_mets_status: Optional[BrainMetsStatus] = None
    measurable_disease_required: Optional[bool] = None
    organ_function_labs: list[LabRequirement] = Field(default_factory=list)
    unstructured_inclusion: list[str] = Field(default_factory=list)
    unstructured_exclusion: list[str] = Field(default_factory=list)
    parse_confidence: Optional[str] = None
    manual_review_flag: bool = False

    @classmethod
    def from_extracted(
        cls, nct_id: str, trial_title: str, overall_status: str, extracted: TrialCriteriaExtracted
    ) -> "TrialCriteria":
        return cls(
            nct_id=nct_id,
            trial_title=trial_title,
            overall_status=overall_status,
            manual_review_flag=bool(extracted.unstructured_inclusion or extracted.unstructured_exclusion),
            **extracted.demographics.model_dump(),
            **extracted.disease.model_dump(),
            **extracted.therapy_history.model_dump(),
            biomarkers=extracted.biomarkers,
            organ_function_labs=extracted.organ_function_labs,
            unstructured_inclusion=extracted.unstructured_inclusion,
            unstructured_exclusion=extracted.unstructured_exclusion,
            parse_confidence=extracted.parse_confidence,
        )


class PatientBiomarker(BaseModel):
    gene: BiomarkerGene
    status: PatientBiomarkerStatus
    detail: Optional[str] = None


class PatientProfile(BaseModel):
    age_years: float
    sex: Sex
    ecog: Optional[int] = None
    histology: Optional[str] = None
    stage: Optional[str] = None
    biomarkers: list[PatientBiomarker] = Field(default_factory=list)
    pdl1_tps: Optional[float] = None
    prior_treatment_history_provided: bool = Field(
        default=False,
        description="True if the user filled in prior-treatment fields at all, to distinguish "
        "'confirmed no prior therapy' from 'unknown/not entered'",
    )
    prior_therapy_lines: Optional[int] = None
    prior_treatments_received: list[str] = Field(default_factory=list)
    brain_mets_status: BrainMetsStatus = BrainMetsStatus.UNKNOWN
    measurable_disease: Optional[bool] = None
    organ_function_labs: dict[str, float] = Field(default_factory=dict)


class CriterionResultEnum(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class CriterionEvaluation(BaseModel):
    field_name: str
    requirement_description: str
    patient_value: Optional[str] = None
    result: CriterionResultEnum


class MatchStatus(str, Enum):
    LIKELY_ELIGIBLE = "Likely Eligible"
    NEEDS_REVIEW = "Possibly Eligible - Needs Review"
    NOT_ELIGIBLE = "Not Eligible"


class MatchResult(BaseModel):
    nct_id: str
    trial_title: str
    overall_status: str
    match_status: MatchStatus
    passed_criteria: list[CriterionEvaluation] = Field(default_factory=list)
    failed_criteria: list[CriterionEvaluation] = Field(default_factory=list)
    unknown_criteria: list[CriterionEvaluation] = Field(default_factory=list)
    manual_review_items: list[str] = Field(default_factory=list)
    explanation: str
    ctgov_url: str
