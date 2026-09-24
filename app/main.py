"""
app/main.py
───────────
Streamlit UI — calls the FastAPI backend at localhost:8000.
Run separately:  streamlit run app/main.py
(Start the API first: uvicorn app.api:app --port 8000)
"""

import os

import httpx
import streamlit as st

# When running via docker-compose the UI container sets API_BASE=http://api:8000
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Agentic RAG Document Assistant")
st.caption("Hybrid retrieval · Cross-encoder reranking · Corrective RAG · Web fallback")


def render_trace(trace: dict):
    """Render transparent execution telemetry without exposing internal prompts or chain-of-thought."""
    if not trace:
        return
    with st.expander("🔍 How this answer was produced"):
        route = trace.get("route", "")
        if route == "csv_data":
            st.markdown("**Route:** 📊 Structured Data (CSV)")
        elif route == "documents":
            st.markdown("**Route:** 📄 Document RAG")
        elif route == "llm_only":
            st.markdown("**Route:** 🧠 Direct LLM Knowledge")
        else:
            st.markdown(f"**Route:** `{route}`")

        # Nodes executed
        st.markdown("**Execution Path:**")
        nodes = trace.get("nodes", [])
        if nodes:
            for n in nodes:
                icon = "✓" if n.get("status") == "success" else "✗"
                err_text = f" — Error: `{n.get('error')}`" if n.get("error") else ""
                st.markdown(f"{icon} `{n.get('name')}` ({n.get('duration_ms', 0)} ms){err_text}")

        # CSV telemetry
        if trace.get("tabular"):
            tab = trace["tabular"]
            st.markdown("---")
            st.markdown("**Structured Data Details:**")
            if tab.get("source"):
                st.write(f"- **Dataset:** `{tab['source']}`")
            op = tab.get("operation", "unknown")
            if tab.get("target_column"):
                op += f"({tab['target_column']})"
            st.write(f"- **Operation:** `{op}`")
            if tab.get("rows_analyzed") is not None:
                st.write(f"- **Rows Analyzed:** {tab['rows_analyzed']}")
            if tab.get("result_count") is not None:
                st.write(f"- **Result Count:** {tab['result_count']}")

        # Document RAG telemetry
        if trace.get("retrieval"):
            ret = trace["retrieval"]
            st.markdown("---")
            st.markdown("**Retrieval Details:**")
            c1, c2 = st.columns(2)
            c1.write(f"- **FAISS candidates:** {ret.get('faiss_candidates', 20)}")
            c1.write(f"- **BM25 candidates:** {ret.get('bm25_candidates', 20)}")
            c2.write(f"- **Reranked:** {ret.get('reranked_count', 0)}")
            c2.write(f"- **Web fallback:** {'Yes' if ret.get('web_fallback') else 'No'}")

        # Sources
        if trace.get("sources"):
            st.markdown("---")
            st.markdown("**Sources Used:**")
            for s in trace["sources"]:
                if s.get("type") == "tabular":
                    st.write(f"- 📊 `{s.get('source')}`: {s.get('summary', '')}")
                elif s.get("type") == "document":
                    st.write(f"- 📄 `{s.get('source')}` — Page {s.get('page', 0)}")
                elif s.get("type") == "web":
                    st.write(f"- 🌐 `{s.get('source')}`")
                else:
                    st.write(f"- 🧠 `{s.get('source')}`")

        tot = trace.get("total_duration_ms")
        if tot:
            st.caption(f"⏱️ Total Execution Time: {tot} ms")


# ── Sidebar — upload ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Knowledge Sources")

    # Document Corpus (PDF)
    st.subheader("📄 Document Corpus (PDF)")
    uploaded_pdf = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_pdf and st.button("Index PDF Document"):
        with st.spinner("Ingesting and indexing PDF …"):
            resp = httpx.post(
                f"{API_BASE}/ingest",
                files={"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")},
                timeout=120,
            )
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Indexed {data.get('chunks', 0)} chunks from {data['filename']}")
        else:
            st.error(f"Error: {resp.text}")

    st.divider()

    # Structured Tabular Data (CSV)
    st.subheader("📊 Structured Data (CSV)")
    uploaded_csv = st.file_uploader("Choose a CSV", type="csv")
    if uploaded_csv and st.button("Load CSV Dataset"):
        with st.spinner("Loading and validating CSV …"):
            resp = httpx.post(
                f"{API_BASE}/ingest/csv",
                files={"file": (uploaded_csv.name, uploaded_csv.getvalue(), "text/csv")},
                timeout=30,
            )
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Loaded {data['filename']} ({data['rows']} rows, {len(data['columns'])} cols)")
            with st.expander("📋 Columns"):
                st.write(", ".join(data['columns']))
        else:
            st.error(f"Error: {resp.text}")

    st.divider()
    st.header("📈 Reliability & Evaluation")
    if st.button("Load Evaluation Report"):
        try:
            resp = httpx.get(f"{API_BASE}/evaluation/results", timeout=15)
            if resp.status_code == 200:
                st.session_state.eval_report = resp.json()
            else:
                st.session_state.eval_report = None
        except Exception as e:
            st.error(f"Cannot load evaluation: {e}")

    if st.session_state.get("eval_report"):
        ev = st.session_state.eval_report
        if "detail" in ev:
            st.info(ev["detail"])
        elif "aggregate" in ev:
            agg = ev["aggregate"]
            st.markdown(f"**Benchmark Dataset:** {ev.get('dataset_size', 0)} cases")
            if agg.get("aggregate_reliability") is not None:
                st.metric("⭐ Aggregate Reliability", f"{agg['aggregate_reliability'] * 100:.1f}%")

            # Document RAG
            st.markdown("##### 📄 Document RAG")
            for k, v in agg.get("document_rag", {}).items():
                if v is not None:
                    st.metric(k.replace("_", " ").title(), f"{v:.3f}")
                else:
                    st.caption(f"{k.replace('_', ' ').title()}: *No Judge*")

            # Retrieval
            st.markdown("##### 🔍 Retrieval")
            ret = agg.get("retrieval", {})
            if ret.get("source_hit_rate") is not None:
                st.metric("Source Hit Rate", f"{ret['source_hit_rate'] * 100:.1f}%")
            if ret.get("retrieval_success_rate") is not None:
                st.metric("Retrieval Success Rate", f"{ret['retrieval_success_rate'] * 100:.1f}%")

            # Tabular
            st.markdown("##### 📊 Structured Data")
            tab = agg.get("structured_data", {})
            if tab.get("correctness_rate") is not None:
                st.metric("CSV Correctness", f"{tab['correctness_rate'] * 100:.1f}%")
            if tab.get("operation_selection_rate") is not None:
                st.metric("Operation Selection", f"{tab['operation_selection_rate'] * 100:.1f}%")

            # Per question table
            with st.expander("📋 Per-Case Breakdown"):
                table_rows = []
                for q in ev.get("results", []):
                    route = q.get("actual_route") or q.get("expected_route", "-")
                    if q.get("tabular_eval"):
                        res_label = "✓ Correct" if q["tabular_eval"].get("overall_correct") else "✗ Incorrect"
                        key_m = f"Op: {q['tabular_eval'].get('operation_selected')}"
                    elif q.get("retrieval_eval"):
                        res_label = "✓ Hit" if q["retrieval_eval"].get("source_hit") else "✗ Miss"
                        key_m = f"Chunks: {len(q['retrieval_eval'].get('retrieved_sources', []))}"
                    else:
                        res_label = q.get("status", "Unknown")
                        key_m = "-"
                    table_rows.append({
                        "Case": q.get("id"),
                        "Route": route,
                        "Result": res_label,
                        "Metric": key_m,
                    })
                if table_rows:
                    st.dataframe(table_rows, use_container_width=True)
    else:
        st.info("Evaluation has not been run. Run `python -m evaluation.evaluate` first.")

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    if str(s.get("source", "")).endswith(".csv"):
                        st.markdown(f"- 📊 **[CSV Data] {s['source']}** — {s.get('text', '')}")
                    else:
                        st.markdown(f"- 📄 **[PDF Document] {s['source']}** — Page {s.get('page', 0)}")
        if msg.get("trace"):
            render_trace(msg["trace"])
        if msg.get("meta"):
            m = msg["meta"]
            cols = st.columns(3)
            cols[0].metric("Latency", f"{m['latency_ms']} ms")
            cols[1].metric("Rewrites", m['rewrite_count'])
            cols[2].metric("Web used", "Yes" if m['used_web'] else "No")

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            try:
                resp = httpx.post(
                    f"{API_BASE}/query",
                    json={
                        "question": prompt,
                        "history": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[:-1]
                        ],
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(data["answer"])

                    if data.get("sources"):
                        with st.expander("📚 Sources"):
                            for s in data["sources"]:
                                if str(s.get("source", "")).endswith(".csv"):
                                    st.markdown(f"- 📊 **[CSV Data] {s['source']}** — {s.get('text', '')}")
                                else:
                                    st.markdown(f"- 📄 **[PDF Document] {s['source']}** — Page {s.get('page', 0)}")

                    if data.get("trace"):
                        render_trace(data["trace"])

                    cols = st.columns(3)
                    cols[0].metric("Latency",   f"{data['latency_ms']} ms")
                    cols[1].metric("Rewrites",  data["rewrite_count"])
                    cols[2].metric("Web used",  "Yes" if data["used_web"] else "No")

                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                        "trace":   data.get("trace"),
                        "meta":    {
                            "latency_ms":    data["latency_ms"],
                            "rewrite_count": data["rewrite_count"],
                            "used_web":      data["used_web"],
                        },
                    })
                else:
                    err = resp.json().get("detail", resp.text)
                    st.error(f"API error: {err}")
            except httpx.ConnectError:
                st.error("Cannot reach API. Start it with: uvicorn app.api:app --port 8000")
