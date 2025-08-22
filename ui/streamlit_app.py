# ui/streamlit_app.py

import os, sys
sys.path.append(os.path.abspath(".")) 

import time
import streamlit as st

# make local package imports work when running from project root
import sys
sys.path.append(os.path.abspath("."))

from app.rag import ingest, rag_answer
from app.agent import run_agent
from app.settings import settings

st.set_page_config(page_title="Agentic RAG", page_icon="📚", layout="wide")

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
k = st.sidebar.number_input("Top K (retrieval)", min_value=1, max_value=10, value=settings.top_k, step=1)
st.sidebar.caption(f"Models: chat={settings.chat_model}, embed={settings.embed_model}")
st.sidebar.write("---")

st.sidebar.subheader("📥 Ingest")
if st.sidebar.button("Rebuild index (ingest docs)", use_container_width=True):
    with st.spinner("Ingesting..."):
        out = ingest()
    st.sidebar.success(out.get("message", out))

uploaded = st.sidebar.file_uploader("Add files (PDF/TXT/MD)", type=["pdf", "txt", "md"], accept_multiple_files=True)
if uploaded:
    os.makedirs(settings.docs_path, exist_ok=True)
    for f in uploaded:
        with open(os.path.join(settings.docs_path, f.name), "wb") as w:
            w.write(f.read())
    st.sidebar.success(f"Saved {len(uploaded)} file(s). Click 'Rebuild index'.")

st.sidebar.write("---")
session_id = st.sidebar.text_input("Session ID (memory key)", value="demo")

# --- Tabs ---
tab1, tab2 = st.tabs(["💬 RAG Q&A", "🤖 Agent (tools + memory)"])

with tab1:
    st.header("RAG Q&A with Citations")
    q = st.text_input("Ask a question about your docs")
    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Thinking..."):
            start = time.time()
            # temporarily override top_k without editing env
            old_k = settings.top_k
            settings.top_k = k
            try:
                out = rag_answer(q)
            finally:
                settings.top_k = old_k
            dur = time.time() - start

        st.success(f"Answered in {dur:.2f}s")
        st.markdown("### Answer")
        st.write(out["answer"])
        st.markdown("### Sources")
        for s in out["sources"]:
            tag = s.get("tag", "")
            src = s.get("source", "unknown")
            page = s.get("page", None)
            page_str = f":p{page}" if page is not None else ""
            st.write(f"{tag} {src}{page_str}")

with tab2:
    st.header("Agent with Tools + Memory")
    user_msg = st.text_area("Your instruction", height=100, placeholder="Summarize the docs then compute 22*13")
    if st.button("Run Agent", type="primary"):
        with st.spinner("Agent working..."):
            out = run_agent(user_msg, session_id=session_id)
        st.markdown("### Output")
        st.write(out["output"])
    st.caption("Tools: search_corpus (RAG), python_calc (safe math). Memory persisted per Session ID.")
