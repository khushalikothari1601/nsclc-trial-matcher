"""Streamlit UI: patient intake form + trial match results.

Run with: streamlit run app.py
"""

import streamlit as st

from cache import cache_stats, list_all_criteria
from matcher import match_patient_to_trial
from parse_criteria import DEFAULT_MODEL
from pipeline import fetch_and_parse
from schema import (
    BiomarkerGene,
    BrainMetsStatus,
    MatchStatus,
    PatientBiomarker,
    PatientBiomarkerStatus,
    PatientProfile,
    Sex,
)

st.set_page_config(page_title="NSCLC Trial Matcher", layout="wide")

st.warning(
    "**Prototype tool.** Do not enter real patient information (PHI) -- use only "
    "synthetic or de-identified test profiles. Output is not clinical advice and has "
    "not been validated for clinical use.",
    icon="⚠️",
)

st.title("NSCLC Trial Matcher")
st.caption(
    "Enter a patient profile to see which currently-cached NSCLC trials from "
    "ClinicalTrials.gov the patient likely qualifies for."
)

# Name -> unit collected. matcher.PATIENT_LAB_UNITS only compares thresholds in these units.
COMMON_LABS = {
    "ANC": "10^9/L",
    "Platelets": "10^9/L",
    "Hemoglobin": "g/dL",
    "Creatinine Clearance": "mL/min",
    "Bilirubin": "mg/dL",
    "AST": "U/L",
    "ALT": "U/L",
}

STATUS_ORDER = {
    MatchStatus.LIKELY_ELIGIBLE: 0,
    MatchStatus.NEEDS_REVIEW: 1,
    MatchStatus.NOT_ELIGIBLE: 2,
}

STATUS_RENDER = {
    MatchStatus.LIKELY_ELIGIBLE: st.success,
    MatchStatus.NEEDS_REVIEW: st.warning,
    MatchStatus.NOT_ELIGIBLE: st.error,
}

# --- Sidebar: cache state + refresh -----------------------------------------------
with st.sidebar:
    st.header("Trial data")
    stats = cache_stats()
    st.metric("Trials cached", stats["trial_count"])
    st.caption(f"Last updated: {stats['last_updated'] or 'never'}")
    st.caption(f"Model: {DEFAULT_MODEL}")

    refresh_n = st.number_input("Trials to fetch on refresh", min_value=5, max_value=100, value=20, step=5)
    if st.button("Refresh trial data", use_container_width=True):
        progress = st.progress(0.0, text="Starting...")

        def _on_progress(i, total, nct_id, action):
            progress.progress(i / total, text=f"[{i}/{total}] {nct_id}: {action}")

        with st.spinner("Fetching and parsing trials..."):
            summary = fetch_and_parse(max_trials=int(refresh_n), on_progress=_on_progress)
        progress.empty()
        st.success(
            f"Fetched {summary['fetched']} trials -- {summary['cache_hits']} cache hits, "
            f"{summary['parsed']} newly parsed, {len(summary['errors'])} errors."
        )
        if summary["errors"]:
            with st.expander("Errors"):
                for err in summary["errors"]:
                    st.text(err)
        st.rerun()

    st.caption(
        "For a larger initial batch, run `python pipeline.py --max-trials 300` from "
        "the command line instead of using this button."
    )

# --- Patient intake form ------------------------------------------------------------
st.header("Patient profile")

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=65.0, step=1.0)
        sex = st.selectbox("Sex", options=[s.value for s in Sex], index=2)
        ecog_choice = st.selectbox("ECOG performance status", options=["Unknown", "0", "1", "2", "3", "4"])
    with col2:
        histology = st.selectbox(
            "Histology",
            options=["", "adenocarcinoma", "squamous", "large_cell", "other", "unknown"],
        )
        stage = st.selectbox("Stage", options=["", "IIIA", "IIIB", "IV", "metastatic", "unknown"])
        pdl1_str = st.text_input("PD-L1 TPS % (blank = unknown)", value="")
    with col3:
        measurable_choice = st.selectbox("Measurable disease", options=["Unknown", "Yes", "No"])
        brain_mets_choice = st.selectbox(
            "Brain metastases status",
            options=[s.value for s in BrainMetsStatus],
            index=len(list(BrainMetsStatus)) - 1,
        )

    st.subheader("Biomarkers")
    biomarker_cols = st.columns(3)
    biomarker_inputs = {}
    for i, gene in enumerate(BiomarkerGene):
        with biomarker_cols[i % 3]:
            status = st.radio(
                gene.value,
                options=[s.value for s in PatientBiomarkerStatus],
                index=2,
                horizontal=True,
                key=f"biomarker_{gene.value}",
            )
            biomarker_inputs[gene] = status

    st.subheader("Prior therapy")
    history_provided = st.checkbox("Prior treatment history entered for this patient")
    ph_col1, ph_col2 = st.columns(2)
    with ph_col1:
        prior_lines_str = st.text_input("Number of prior systemic therapy lines", value="")
    with ph_col2:
        prior_drugs_str = st.text_input("Prior treatments received (comma-separated)", value="")

    st.subheader("Organ function labs (optional)")
    lab_cols = st.columns(4)
    lab_inputs = {}
    for i, (lab_name, unit) in enumerate(COMMON_LABS.items()):
        with lab_cols[i % 4]:
            lab_inputs[lab_name] = st.text_input(f"{lab_name} ({unit})", value="", key=f"lab_{lab_name}")

    submitted = st.form_submit_button("Find matching trials", use_container_width=True)


def _parse_optional_float(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _build_patient_profile() -> PatientProfile:
    biomarkers = [
        PatientBiomarker(gene=gene, status=PatientBiomarkerStatus(status))
        for gene, status in biomarker_inputs.items()
    ]

    organ_labs = {}
    for lab_name, raw in lab_inputs.items():
        val = _parse_optional_float(raw)
        if val is not None:
            organ_labs[lab_name] = val

    prior_lines = None
    parsed_lines = _parse_optional_float(prior_lines_str)
    if parsed_lines is not None:
        prior_lines = int(parsed_lines)

    prior_drugs = [d.strip() for d in prior_drugs_str.split(",") if d.strip()]

    return PatientProfile(
        age_years=age,
        sex=Sex(sex),
        ecog=None if ecog_choice == "Unknown" else int(ecog_choice),
        histology=histology or None,
        stage=stage or None,
        biomarkers=biomarkers,
        pdl1_tps=_parse_optional_float(pdl1_str),
        prior_treatment_history_provided=history_provided,
        prior_therapy_lines=prior_lines,
        prior_treatments_received=prior_drugs,
        brain_mets_status=BrainMetsStatus(brain_mets_choice),
        measurable_disease={"Yes": True, "No": False, "Unknown": None}[measurable_choice],
        organ_function_labs=organ_labs,
    )


# --- Results -------------------------------------------------------------------------
if submitted:
    patient = _build_patient_profile()
    trials = list_all_criteria()

    if not trials:
        st.info("No trials cached yet. Use 'Refresh trial data' in the sidebar, or run pipeline.py first.")
    else:
        results = [match_patient_to_trial(patient, trial) for trial in trials]
        results.sort(key=lambda r: STATUS_ORDER[r.match_status])

        st.header(f"Results ({len(results)} trials evaluated)")

        status_filter = st.multiselect(
            "Filter by status",
            options=[s.value for s in MatchStatus],
            default=[MatchStatus.LIKELY_ELIGIBLE.value, MatchStatus.NEEDS_REVIEW.value],
            help="'Not Eligible' is excluded by default. Real trials almost always have some "
            "free-text eligibility criteria that can't be structured, so most real matches "
            "land in 'Needs Review' rather than 'Likely Eligible' -- check the flagged "
            "criteria yourself before assuming a trial applies.",
        )
        results = [r for r in results if r.match_status.value in status_filter]

        if not results:
            st.info(
                "No trials match the current filter. Try adding 'Possibly Eligible - "
                "Needs Review' above, or fetch more trials from the sidebar."
            )

        for result in results:
            render_fn = STATUS_RENDER[result.match_status]
            with st.expander(f"{result.match_status.value} -- {result.trial_title or result.nct_id} ({result.nct_id})"):
                render_fn(result.match_status.value)
                st.caption(result.explanation)
                st.markdown(f"[View on ClinicalTrials.gov]({result.ctgov_url})")

                if result.failed_criteria:
                    st.markdown("**Failed criteria**")
                    st.table(
                        [
                            {"Criterion": e.field_name, "Requirement": e.requirement_description, "Patient value": e.patient_value}
                            for e in result.failed_criteria
                        ]
                    )
                if result.unknown_criteria:
                    st.markdown("**Needs patient data**")
                    st.table(
                        [
                            {"Criterion": e.field_name, "Requirement": e.requirement_description}
                            for e in result.unknown_criteria
                        ]
                    )
                if result.passed_criteria:
                    st.markdown("**Passed criteria**")
                    st.table(
                        [
                            {"Criterion": e.field_name, "Requirement": e.requirement_description, "Patient value": e.patient_value}
                            for e in result.passed_criteria
                        ]
                    )
                if result.manual_review_items:
                    st.markdown("**Manual review needed (not captured by structured fields)**")
                    for item in result.manual_review_items:
                        st.markdown(f"- {item}")
