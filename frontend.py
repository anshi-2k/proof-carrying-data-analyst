import os

import streamlit as st

from dataproof_engine import answer_question, load_tables
from cleaning import clean_tables
from guardrails import analyze_tables

st.set_page_config(page_title="DataProof AI", page_icon="🔍", layout="wide")
st.title("🔍 DataProof AI")
st.caption("Verified answers with re-runnable proof. It refuses when the data can't support an answer.")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

with st.sidebar:
    st.header("📂 Tables")
    files = st.file_uploader("Upload one or more CSV / Excel files", type=["csv", "xlsx"],
                             accept_multiple_files=True)

if not files:
    st.info("Upload one or more tables from the sidebar to begin.")
    st.stop()

paths = []
for f in files:
    p = os.path.join(UPLOAD_DIR, f.name)
    with open(p, "wb") as out:
        out.write(f.getbuffer())
    paths.append(p)

raw = load_tables(paths)
cleaned, clog = clean_tables(raw)
findings = analyze_tables(raw, cleaned)

ICON = {"blocking": "⛔", "warning": "⚠️", "info": "ℹ️"}
st.header("📊 Data checks")
tabs = st.tabs(list(raw))
for tab, (name, df) in zip(tabs, raw.items()):
    with tab:
        st.caption(f"{len(df)} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True)
for f in findings:
    st.markdown(f"{ICON[f['severity']]} {f['message']}")
if not findings:
    st.success("No issues detected.")

st.header("💬 Ask your data")
question = st.text_input("Question", placeholder="What is the total quantity sold?")
if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()
    with st.spinner("Analyzing..."):
        res = answer_question(question, paths)

    if res["status"] == "ANSWER":
        st.success("✅ Answer")
        st.write(res["result"])
        for a in res["assumptions"]:
            st.info(f"Assumption: {a}")
        st.subheader("🔎 Proof")
        st.code(res["code"], language="python")
        st.success("Independently re-run in a fresh process: same result." if res["verified"]
                   else "Not re-run.")
        if res["proof_script"]:
            with open(res["proof_script"], encoding="utf-8") as fh:
                st.download_button("⬇ Download re-runnable proof script", fh.read(),
                                   file_name=os.path.basename(res["proof_script"]))
    elif res["status"] == "NEEDS_CLARIFICATION":
        st.warning("❓ I need a clarification before answering")
        st.write(res["reason"])
    else:
        st.error("🚫 I can't determine this")
        st.write(res["reason"])
