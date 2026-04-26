"""
app.py — Resume Intelligence Streamlit Application
"""

from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import streamlit as st

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Resume Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy pipeline import ────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models — this takes ~10 s on first run…")
def _load_pipeline():
    from resume_intelligence.pipeline.pipeline import run_pipeline
    return run_pipeline


# ── Sample Data ─────────────────────────────────────────
_SAMPLE_RESUME = """Alex Johnson
Data Scientist with 4 years of experience in machine learning at Acme Corp.
Skills: Python, machine learning, pandas, scikit-learn, SQL, TensorFlow, Docker
Education: M.S. Computer Science - Stanford University (2020)
"""

_SAMPLE_JD = """Senior Data Scientist
Looking for Python, machine learning, pandas, scikit-learn, SQL, TensorFlow, Docker, AWS, Kubernetes.
4+ years required.
"""


# ── File Extraction ─────────────────────────────────────
def _extract_text_from_file(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if name.endswith(".txt"):
        return raw_bytes.decode("utf-8", errors="replace")

    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw_bytes))
        return "\n".join([p.extract_text() or "" for p in reader.pages])

    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(raw_bytes))
        return "\n".join([p.text for p in doc.paragraphs])

    return None


# ── Sidebar ─────────────────────────────────────────────
def _render_sidebar():
    st.sidebar.title("⚙️ Settings")

    use_sample = st.sidebar.checkbox("Use Sample Data")
    run_audit = st.sidebar.checkbox("Run Bias Audit", value=True)

    return use_sample, run_audit


# ── Inputs ──────────────────────────────────────────────
def _render_inputs(use_sample):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Resume")
        uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
        resume_text = st.text_area("Paste Resume")

    with col2:
        st.subheader("💼 Job Description")
        uploaded_jd = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"])
        jd_text = st.text_area("Paste JD")

    if uploaded_resume:
        resume_text = _extract_text_from_file(uploaded_resume)

    if uploaded_jd:
        jd_text = _extract_text_from_file(uploaded_jd)

    if use_sample:
        resume_text = _SAMPLE_RESUME
        jd_text = _SAMPLE_JD

    return resume_text, jd_text


# ── Main ────────────────────────────────────────────────
def main():
    st.title("📄 Resume Intelligence")

    use_sample, run_audit = _render_sidebar()
    resume_text, jd_text = _render_inputs(use_sample)

    if st.button("🚀 Analyze Resume"):

        if not resume_text or not jd_text:
            st.error("Provide both resume and job description")
            return

        run_pipeline = _load_pipeline()

        with st.spinner("Running pipeline..."):
            result = run_pipeline(
                resume_text=resume_text,
                job_text=jd_text,
                run_audit=run_audit
            )

        st.success("Analysis Complete ✅")

        st.write("### Score:", result.scoring_result.final_score)
        st.write("Matched Skills:", result.scoring_result.matched_skills)
        st.write("Missing Skills:", result.scoring_result.missing_skills)

        if run_audit:
            st.write("### Bias Audit")
            st.write(result.fairness_report.summary)


if __name__ == "__main__":
    main()
