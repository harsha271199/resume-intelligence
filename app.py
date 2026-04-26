"""
app.py — Resume Intelligence Streamlit Application

Run with:
    streamlit run app.py

Sections
--------
A. Resume Parser   — parse raw resume text → structured entities
B. Job Matching    — score resume against job description
C. Bias Audit      — name-swapping fairness experiment
"""

from __future__ import annotations

import io
import os
import sys
from typing import Optional

# ── Robust src path injection ─────────────────────────────────────────────────
# MUST be before any resume_intelligence imports.
# os.path.abspath(__file__) works on both local dev and Streamlit Cloud
# regardless of the current working directory.

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_PATH = os.path.join(_ROOT_DIR, "src")

if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Resume Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy pipeline import (avoids slow model load on every rerun) ──────────────
@st.cache_resource(show_spinner="Loading AI models — this takes ~10 s on first run…")
def _load_pipeline():
    from resume_intelligence.pipeline.pipeline import run_pipeline
    return run_pipeline


# ── Sample data ───────────────────────────────────────────────────────────────

_SAMPLE_RESUME = """\
Alex Johnson
Email: alex.johnson@email.com

SUMMARY
Data Scientist with 4 years of experience in machine learning and NLP at Acme Corp.

SKILLS
Python, machine learning, NLP, pandas, scikit-learn, SQL, TensorFlow, Docker, AWS

EXPERIENCE
Data Scientist — Acme Corp (2020–2024)
- Built ML pipelines for customer churn prediction using scikit-learn
- Developed NLP models for sentiment analysis with TensorFlow
- 4 years of experience in machine learning and data analysis

EDUCATION
M.S. Computer Science — Stanford University (2020)
B.S. Statistics — UC Berkeley (2018)
"""

_SAMPLE_JD = """\
Senior Data Scientist — TechCorp

We are looking for a Senior Data Scientist to join our AI team.

Requirements:
- 4+ years of experience in machine learning and data science
- Strong Python skills with pandas, scikit-learn, and TensorFlow
- Experience with SQL, Docker, and AWS
- Knowledge of NLP and deep learning is a plus
- Kubernetes experience preferred

Responsibilities:
- Build and deploy ML models at scale
- Collaborate with product and engineering teams
- Mentor junior data scientists
"""


# ── File text extraction ──────────────────────────────────────────────────────

def _extract_text_from_file(uploaded_file) -> Optional[str]:
    """
    Extract plain text from an uploaded file.
    Supports .txt, .pdf (pypdf), .docx (python-docx).
    """
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if name.endswith(".txt"):
        return raw_bytes.decode("utf-8", errors="replace")

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages).strip()
            if not text:
                st.warning("PDF has no extractable text layer. Try a text-based PDF.")
            return text
        except Exception as exc:
            st.error(f"Could not read PDF: {exc}")
            return None

    if name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(raw_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except Exception as exc:
            st.error(f"Could not read DOCX: {exc}")
            return None

    st.warning(f"Unsupported file type: {uploaded_file.name}")
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> dict:
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Data")
    use_sample = st.sidebar.checkbox(
        "Use Sample Data",
        value=False,
        help="Pre-fill inputs with a sample Data Scientist resume and job description",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Pipeline Options")
    run_audit = st.sidebar.checkbox(
        "Run Bias Audit",
        value=True,
        help="Name-swapping fairness experiment — adds ~5–10 s",
    )
    show_tfidf = st.sidebar.checkbox(
        "Show TF-IDF Baseline",
        value=False,
        help="Compare neural score against keyword-matching baseline",
    )
    show_raw_json = st.sidebar.checkbox(
        "Show Raw JSON",
        value=False,
        help="Expand full parsed output as JSON for debugging",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "**Resume Intelligence**\n\n"
        "A Multi-Agent Framework for Structured and Fair Resume Screening.\n\n"
        "**Agents:**\n"
        "- 📋 Resume Parser (spaCy NER)\n"
        "- 🎯 Semantic Scorer (MiniLM)\n"
        "- ⚖️ Bias Auditor (name-swapping)"
    )

    return {
        "use_sample": use_sample,
        "run_audit": run_audit,
        "show_tfidf": show_tfidf,
        "show_raw_json": show_raw_json,
    }


# ── Input widgets ─────────────────────────────────────────────────────────────

def _resolve_text(
    file_uploader_key: str,
    text_area_key: str,
    use_sample: bool,
    sample_text: str,
) -> str:
    """
    Inject the correct text into session_state BEFORE the text area renders.

    Priority: uploaded file > sample data toggle > user-typed text.
    Uses session_state injection so the text area reflects file content
    on the same rerun the file was uploaded (st.text_area value= is ignored
    after first render).
    """
    uploaded = st.session_state.get(file_uploader_key)

    if uploaded is not None:
        file_id = (uploaded.name, uploaded.size)
        last_id_key = f"_last_file_{file_uploader_key}"
        if st.session_state.get(last_id_key) != file_id:
            extracted = _extract_text_from_file(uploaded)
            if extracted and extracted.strip():
                st.session_state[text_area_key] = extracted
            st.session_state[last_id_key] = file_id
    elif use_sample:
        if not st.session_state.get(text_area_key, "").strip():
            st.session_state[text_area_key] = sample_text

    return st.session_state.get(text_area_key, "")


def _render_inputs(use_sample: bool) -> tuple[str, str]:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Resume")
        st.file_uploader(
            "Upload resume (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            key="resume_file",
        )
        _resolve_text("resume_file", "resume_text", use_sample, _SAMPLE_RESUME)

        if st.session_state.get("resume_file") is not None:
            st.caption(f"📎 Using uploaded file: **{st.session_state['resume_file'].name}**")
        elif use_sample and st.session_state.get("resume_text", "").strip() == _SAMPLE_RESUME.strip():
            st.caption("📋 Using sample data")

        resume_text = st.text_area(
            "Or paste resume text directly",
            height=320,
            placeholder="Paste resume here...",
            key="resume_text",
        )

    with col2:
        st.subheader("💼 Job Description")
        st.file_uploader(
            "Upload job description (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            key="jd_file",
        )
        _resolve_text("jd_file", "jd_text", use_sample, _SAMPLE_JD)

        if st.session_state.get("jd_file") is not None:
            st.caption(f"📎 Using uploaded file: **{st.session_state['jd_file'].name}**")
        elif use_sample and st.session_state.get("jd_text", "").strip() == _SAMPLE_JD.strip():
            st.caption("📋 Using sample data")

        jd_text = st.text_area(
            "Or paste job description directly",
            height=320,
            placeholder="Paste job description here...",
            key="jd_text",
        )

    return resume_text, jd_text


# ── Pipeline status banner ────────────────────────────────────────────────────

def _render_pipeline_status(agents_completed: list) -> None:
    agents = [
        ("parsing",  "📋 Resume Parser"),
        ("scoring",  "🎯 Job Matching"),
        ("auditing", "⚖️ Bias Audit"),
    ]
    cols = st.columns(len(agents))
    for col, (key, label) in zip(cols, agents):
        if key in agents_completed:
            col.success(f"✅ {label}")
        else:
            col.error(f"❌ {label}")


# ── Section A: Resume Parser ──────────────────────────────────────────────────

def _render_parser_section(result, show_raw: bool) -> None:
    st.markdown("---")
    st.subheader("📋 Resume Parser")

    pr = result.parsed_resume

    if pr.parse_error:
        st.warning(f"⚠️ Parser warning: {pr.parse_error}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Skills", len(pr.normalized_skills))
    m2.metric("Education", len(pr.education))
    m3.metric("Experience", len(pr.experience))
    m4.metric("Organizations", len(pr.organizations))

    if pr.name:
        st.markdown(f"**👤 Candidate detected:** {pr.name}")

    c1, c2 = st.columns(2)

    with c1:
        if pr.normalized_skills:
            with st.expander(f"✅ Extracted Skills ({len(pr.normalized_skills)})", expanded=True):
                st.markdown(" ".join(f"`{s}`" for s in sorted(pr.normalized_skills)))
        if pr.job_titles:
            with st.expander(f"💼 Job Titles ({len(pr.job_titles)})"):
                for t in pr.job_titles:
                    st.markdown(f"- {t.title()}")

    with c2:
        if pr.education:
            with st.expander(f"🎓 Education ({len(pr.education)})", expanded=True):
                for e in pr.education:
                    st.markdown(f"- {e}")
        if pr.organizations:
            with st.expander(f"🏢 Organizations ({len(pr.organizations)})"):
                for o in pr.organizations:
                    st.markdown(f"- {o}")

    if show_raw:
        with st.expander("🔍 Raw JSON — ParsedResume"):
            d = pr.model_dump()
            d.pop("raw_text", None)
            st.json(d)


# ── Section B: Job Matching ───────────────────────────────────────────────────

def _score_badge(score: float) -> str:
    if score >= 75:
        return "🟢"
    if score >= 50:
        return "🟡"
    return "🔴"


def _render_scoring_section(result, show_tfidf: bool, show_raw: bool) -> None:
    st.markdown("---")
    st.subheader("🎯 Job Matching")

    sr = result.scoring_result
    jd = result.job_description

    if sr.score_error:
        st.error(f"Scoring error: {sr.score_error}")
        return

    badge = _score_badge(sr.final_score)
    st.markdown(
        f"### {badge} Skill Alignment Score: **{sr.final_score:.1f} / 100**"
        f"  —  *{jd.job_title}*"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Semantic Similarity", f"{sr.semantic_similarity:.1f}")
    c2.metric("Skill Coverage", f"{sr.skill_coverage:.1f}")
    c3.metric("Experience Match", f"{sr.experience_match:.1f}")
    c4.metric("Skill Gap Penalty", f"{sr.skill_gap_penalty:.1f}")

    if show_tfidf and sr.tfidf_baseline_score is not None:
        st.caption(
            f"TF-IDF baseline: **{sr.tfidf_baseline_score:.1f}** / 100  "
            f"(neural adds {sr.final_score - sr.tfidf_baseline_score:+.1f} pts)"
        )

    col1, col2 = st.columns(2)

    with col1:
        if sr.matched_skills:
            with st.expander(f"✅ Matched Skills ({len(sr.matched_skills)})", expanded=True):
                st.markdown(" ".join(f"`{s}`" for s in sr.matched_skills))
        if sr.semantic_matches:
            with st.expander(f"🔍 Semantic Matches ({len(sr.semantic_matches)})"):
                st.markdown(" ".join(f"`{s}`" for s in sr.semantic_matches))

    with col2:
        if sr.missing_skills:
            with st.expander(f"❌ Missing Skills ({len(sr.missing_skills)})", expanded=True):
                st.markdown(" ".join(f"`{s}`" for s in sr.missing_skills))
        else:
            st.success("All required skills are covered!")

    with st.expander("💡 Explanation & Suggestions", expanded=True):
        for line in sr.explanation.split("\n"):
            if line.strip():
                st.markdown(line)

    if show_raw:
        with st.expander("🔍 Raw JSON — ScoringResult"):
            st.json(sr.model_dump())


# ── Section C: Bias Audit ─────────────────────────────────────────────────────

def _render_audit_section(result, show_raw: bool) -> None:
    st.markdown("---")
    st.subheader("⚖️ Bias Audit")

    fr = result.fairness_report
    if fr is None:
        st.info("Bias audit was skipped (disabled in sidebar).")
        return

    if fr.audit_error:
        st.error(f"Audit error: {fr.audit_error}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Score", f"{fr.original_score:.1f}")
    c2.metric("Max Score Shift", f"{fr.max_difference:.2f} pts")
    c3.metric("Demographic Parity Diff", f"{fr.demographic_parity_difference:.4f}")

    if fr.max_difference < 2.0:
        st.success(f"✅ {fr.summary}")
    elif fr.max_difference < 5.0:
        st.info(f"ℹ️ {fr.summary}")
    elif fr.max_difference < 10.0:
        st.warning(f"⚠️ {fr.summary}")
    else:
        st.error(f"🚨 {fr.summary}")

    if fr.swapped_scores:
        with st.expander("📊 Per-Name Score Table", expanded=True):
            original = fr.original_score
            rows = [
                {
                    "Name": name,
                    "Score": round(score, 2),
                    "Shift (pts)": round(abs(score - original), 2),
                    "Direction": "▲ Higher" if score > original else ("▼ Lower" if score < original else "= Same"),
                }
                for name, score in sorted(
                    fr.swapped_scores.items(), key=lambda x: x[1], reverse=True
                )
            ]
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

    if show_raw:
        with st.expander("🔍 Raw JSON — FairnessReport"):
            st.json(fr.model_dump())


# ── Landing page ──────────────────────────────────────────────────────────────

def _render_landing() -> None:
    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📋 Step 1 — Parse**")
        st.markdown(
            "spaCy NER + keyword matching extracts skills, education, "
            "experience, organisations, and job titles from raw resume text."
        )
    with c2:
        st.markdown("**🎯 Step 2 — Score**")
        st.markdown(
            "MiniLM sentence embeddings compute semantic similarity, "
            "combined with skill coverage and experience match into a "
            "Skill Alignment Score (0–100)."
        )
    with c3:
        st.markdown("**⚖️ Step 3 — Audit**")
        st.markdown(
            "Controlled name-swapping: the same resume is re-scored with "
            "10+ demographically distinct names. Score shifts reveal "
            "potential bias in the scoring model."
        )
    st.markdown("---")
    st.caption(
        "Paste text directly, upload a file (PDF / DOCX / TXT), "
        "or enable **Use Sample Data** in the sidebar, then click **Analyze Resume**."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📄 Resume Intelligence")
    st.markdown(
        "A Multi-Agent Framework for **Structured** and **Fair** Resume Screening."
    )
    st.markdown("---")

    options = _render_sidebar()
    resume_text, jd_text = _render_inputs(options["use_sample"])

    st.markdown("")
    analyze = st.button(
        "🚀 Analyze Resume", type="primary", use_container_width=True
    )

    # Read from session_state — always reflects file-uploaded or typed text
    _resume = st.session_state.get("resume_text", resume_text)
    _jd = st.session_state.get("jd_text", jd_text)
    cache_key = (_resume.strip(), _jd.strip(), options["run_audit"])

    if analyze:
        _resume = st.session_state.get("resume_text", resume_text)
        _jd = st.session_state.get("jd_text", jd_text)

        if not _resume.strip():
            st.error("Please provide resume text — paste it, upload a file, or enable Use Sample Data.")
            return
        if not _jd.strip():
            st.error("Please provide a job description — paste it, upload a file, or enable Use Sample Data.")
            return

        if st.session_state.get("_cache_key") != cache_key:
            run_pipeline = _load_pipeline()
            with st.spinner("Running pipeline… this may take 10–20 s on first run."):
                try:
                    result = run_pipeline(
                        resume_text=_resume,
                        job_text=_jd,
                        run_audit=options["run_audit"],
                    )
                    st.session_state["_result"] = result
                    st.session_state["_cache_key"] = cache_key
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    return

        st.success("Analysis Complete ✅")

    if "_result" in st.session_state:
        result = st.session_state["_result"]

        st.markdown("#### Pipeline Status")
        _render_pipeline_status(result.agents_completed)

        _render_parser_section(result, options["show_raw_json"])
        _render_scoring_section(result, options["show_tfidf"], options["show_raw_json"])

        if options["run_audit"]:
            _render_audit_section(result, options["show_raw_json"])
    else:
        _render_landing()


if __name__ == "__main__":
    main()
