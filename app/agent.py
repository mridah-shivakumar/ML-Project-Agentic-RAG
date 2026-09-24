"""
app/agent.py
────────────
Corrective RAG Agent built with LangGraph 0.4+

Graph nodes
───────────
  route_query      → decides: use documents OR answer from LLM knowledge
  retrieve         → hybrid FAISS+BM25 search + cross-encoder rerank
  grade_documents  → LLM judges if retrieved chunks are relevant
  rewrite_query    → rewrites query if grading failed (Corrective RAG loop)
  generate         → produces final answer with citations
  web_search       → Tavily fallback when docs fail after rewrite

State machine
─────────────
  route_query
      ├─→ "documents"  → retrieve → grade_documents
      │                     ├─→ "generate"     → generate → END
      │                     └─→ "rewrite"      → rewrite_query → retrieve (loop, max 2x)
      │                           └─→ "web_search" (after 2 rewrites) → generate → END
      └─→ "llm_only"   → generate → END
"""

from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from loguru import logger

from ingestion.tabular_store import tabular_store
from vectorstore.store import hybrid_search

load_dotenv()

# ── LLM (local Ollama — swap to any LangChain-compatible model) ───────────────
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")   # change to mistral, qwen2.5, etc.
llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

# ── Web search fallback (free Tavily key at tavily.com) ──────────────────────
web_search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
)

MAX_REWRITES = 2   # prevent infinite loops


# ── Agent state ──────────────────────────────────────────────────────────────

class _AgentStateRequired(TypedDict):
    messages:      Annotated[List[BaseMessage], add_messages]   # full conversation history
    query:         str
    rewrite_count: int
    context:       List[Dict[str, Any]]   # retrieved chunk dicts from hybrid_search
    web_results:   List[str]
    answer:        str

class AgentState(_AgentStateRequired, total=False):
    """LangGraph state — _route and _grade are injected by nodes, not part of initial state."""
    _route:          str   # 'documents' | 'llm_only' | 'csv_data'
    _grade:          str   # 'generate' | 'rewrite'
    tabular_context: Optional[Dict[str, Any]]   # structured computation result from tabular_store


# ── Node helpers ─────────────────────────────────────────────────────────────

def _last_human_query(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return state.get("query") or ""  # type: ignore[union-attr]


def _format_history(messages: List[BaseMessage], max_turns: int = 4) -> str:
    """Format recent prior conversation turns (excluding the final current query)."""
    prior = messages[:-1]
    if not prior:
        return ""
    return "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in prior[-max_turns:]
    )


# ── Nodes ────────────────────────────────────────────────────────────────────

def route_query(state: AgentState) -> AgentState:
    """
    Ask LLM whether the question needs document retrieval, tabular CSV analysis, or general knowledge.
    Contextualizes follow-up questions if conversation history exists.
    Returns updated state; routing decision is in '_route' and search query in 'query'.
    """
    query = _last_human_query(state)
    history_text = _format_history(state.get("messages", []))
    has_csv = tabular_store.has_data()
    csv_schema = tabular_store.get_schema_summary() if has_csv else ""

    csv_prompt_section = ""
    csv_choice = ""
    if has_csv:
        csv_prompt_section = (
            f"\nAn active tabular CSV dataset is loaded with schema:\n{csv_schema}\n"
            "- Reply 'csv_data' if the question asks about data, numbers, calculations, statistics, "
            "or columns from the loaded CSV dataset.\n"
        )
        csv_choice = " OR csv_data"

    if history_text:
        prompt = (
            "You are a routing assistant. Given the conversation history and latest question, decide:\n"
            f"{csv_prompt_section}"
            "- Reply 'documents' if the answer likely requires looking up specific text documents/PDFs.\n"
            "- Reply 'llm_only' if it's general knowledge you can answer without documents or tables.\n\n"
            f"Conversation History:\n{history_text}\n\n"
            f"Latest Question: {query}\n\n"
            f"Reply with exactly one word: documents{csv_choice} OR llm_only"
        )
    else:
        prompt = (
            "You are a routing assistant. Given the user question below, decide:\n"
            f"{csv_prompt_section}"
            "- Reply 'documents' if the answer likely requires looking up specific text documents/PDFs.\n"
            "- Reply 'llm_only' if it's general knowledge you can answer without documents or tables.\n\n"
            f"Question: {query}\n\nReply with exactly one word: documents{csv_choice} OR llm_only"
        )

    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().lower()
    if "csv_data" in decision and has_csv:
        decision = "csv_data"
    elif "documents" in decision:
        decision = "documents"
    else:
        decision = "llm_only"
    logger.info(f"[route_query] decision='{decision}' for query='{query[:60]}'")

    # Contextualize follow-up query if history is present and routing to documents or csv_data
    if decision in ("documents", "csv_data") and history_text:
        rephrase_prompt = (
            "Given the conversation history and the user's latest follow-up question, "
            "rephrase the question into a standalone, specific query that incorporates necessary context "
            "(resolving pronouns like 'it', 'they', 'this', 'that', 'which one'). "
            "If the question is already fully standalone, return it unchanged. "
            "Return ONLY the standalone search query without preamble or quotes.\n\n"
            f"Conversation History:\n{history_text}\n\n"
            f"Latest Question: {query}"
        )
        rephrase_resp = llm.invoke([HumanMessage(content=rephrase_prompt)])
        standalone = str(rephrase_resp.content).strip()
        if standalone:
            logger.info(f"[route_query] contextualized query from '{query}' to '{standalone}'")
            query = standalone

    return {**state, "query": query, "_route": decision}


def retrieve(state: AgentState) -> AgentState:
    """Hybrid search → cross-encoder rerank → store chunks in state."""
    query   = state["query"]
    results = hybrid_search(query, top_k=5)
    logger.info(f"[retrieve] got {len(results)} chunks")
    return {**state, "context": results}


def grade_documents(state: AgentState) -> AgentState:
    """
    LLM grades each retrieved chunk for relevance.
    Marks state with '_grade': 'generate' or 'rewrite'.
    """
    query   = state["query"]
    context = state["context"]

    relevant = []
    for chunk in context:
        prompt = (
            f"Question: {query}\n\n"
            f"Document chunk:\n{chunk['text']}\n\n"
            "Is this chunk relevant to answering the question? Reply yes or no."
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        if "yes" in resp.content.lower():
            relevant.append(chunk)

    logger.info(f"[grade_documents] {len(relevant)}/{len(context)} chunks relevant")

    if relevant:
        return {**state, "context": relevant, "_grade": "generate"}
    else:
        return {**state, "_grade": "rewrite"}


def rewrite_query(state: AgentState) -> AgentState:
    """Corrective RAG: rewrite the query to improve retrieval, using conversation context if available."""
    query = state["query"]
    history_text = _format_history(state.get("messages", []))

    if history_text:
        prompt = (
            f"The following question failed to retrieve useful documents from our knowledge base:\n{query}\n\n"
            f"Conversation History:\n{history_text}\n\n"
            "Using the conversation context to clarify any ambiguous references, rewrite the question "
            "to be more specific and likely to match technical documentation. "
            "Return only the rewritten question, nothing else."
        )
    else:
        prompt = (
            f"The following question didn't retrieve useful documents:\n{query}\n\n"
            "Rewrite it to be more specific and likely to match technical documentation. "
            "Return only the rewritten question, nothing else."
        )

    response      = llm.invoke([HumanMessage(content=prompt)])
    new_query     = str(response.content).strip()
    rewrite_count = int(state.get("rewrite_count") or 0) + 1  # type: ignore[union-attr]
    logger.info(f"[rewrite_query] attempt {rewrite_count}: '{new_query[:80]}'")
    return {**state, "query": new_query, "rewrite_count": rewrite_count}


def web_search(state: AgentState) -> AgentState:
    """Tavily web search as last-resort fallback."""
    query   = state["query"]
    results = web_search_tool.invoke(query)
    snippets = [r["content"] for r in results if "content" in r]
    logger.info(f"[web_search] got {len(snippets)} web results")
    return {**state, "web_results": snippets}


def analyze_tabular(state: AgentState) -> AgentState:
    """
    Formulates a strict JSON analysis plan and executes safe pandas operations
    via tabular_store without any eval() or exec().
    """
    query = state["query"]
    if not tabular_store.has_data():
        err_ctx = {
            "status": "error",
            "error": "No CSV dataset is currently loaded. Please upload a CSV file to analyze tabular data."
        }
        return {**state, "tabular_context": err_ctx}

    schema = tabular_store.get_schema_summary()
    plan_prompt = (
        "You are a strict data analysis parameter planner. Given the user query and dataset schema, "
        "produce a single valid JSON object containing the exact parameters to answer the query.\n\n"
        f"Dataset Schema:\n{schema}\n\n"
        f"User Query: {query}\n\n"
        "Allowed Operations:\n"
        "- 'highest': Finds row with maximum value in a numeric column (requires target_column)\n"
        "- 'lowest': Finds row with minimum value in a numeric column (requires target_column)\n"
        "- 'aggregate': Calculates a statistic (requires target_column and agg_func: 'sum'|'mean'|'count'|'min'|'max'|'std'|'median')\n"
        "- 'frequency': Finds most frequent category/value (requires target_column)\n"
        "- 'filter': Returns matching rows (requires filter: {'column': ..., 'operator': '=='|'!='|'>'|'>='|'<'|'<='|'contains', 'value': ...})\n"
        "- 'groupby_agg': Groups by group_column and aggregates target_column (requires group_column, target_column, agg_func)\n"
        "- 'describe': Summary statistics across dataset or target_column\n"
        "- 'list_values': Lists unique values in target_column\n\n"
        "Respond with ONLY a JSON object in this exact schema, with no surrounding markdown or explanation:\n"
        "{\n"
        '  "operation": "highest" | "lowest" | "aggregate" | "frequency" | "filter" | "groupby_agg" | "describe" | "list_values",\n'
        '  "target_column": "<exact column name or null>",\n'
        '  "group_column": "<exact column name or null>",\n'
        '  "agg_func": "sum" | "mean" | "count" | "min" | "max" | "std" | "median" | null,\n'
        '  "filter": {"column": "<col>", "operator": "=="|"!="|">"|">="|"<"|"<="|"contains", "value": "<val>"} or null\n'
        "}"
    )

    try:
        response = llm.invoke([HumanMessage(content=plan_prompt)])
        raw_text = str(response.content).strip()
        if "```" in raw_text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            if match:
                raw_text = match.group(1)
        plan_params = json.loads(raw_text)
    except Exception as e:
        logger.warning(f"[analyze_tabular] JSON extraction failed ({e}), falling back to describe")
        plan_params = {"operation": "describe"}

    logger.info(f"[analyze_tabular] Executing plan: {plan_params}")
    result = tabular_store.execute_query(plan_params)
    logger.info(f"[analyze_tabular] Result status: {result.get('status')}")

    return {**state, "tabular_context": result}


def generate(state: AgentState) -> AgentState:
    """
    Final generation node — synthesises answer from context + conversation history.
    Supports document chunks, web results, and structured tabular computation results.
    """
    query:       str                       = state["query"]
    context:     List[Dict[str, Any]]     = state.get("context") or []
    web_results: List[str]                = state.get("web_results") or []
    tabular_ctx: Optional[Dict[str, Any]] = state.get("tabular_context")
    history:     List[BaseMessage]        = state.get("messages") or []

    # Build context block
    if tabular_ctx:
        if tabular_ctx.get("status") == "success":
            ctx_block = (
                f"[Source: {tabular_ctx.get('source', 'dataset.csv')}]\n"
                f"Operation: {tabular_ctx.get('operation')}\n"
                f"Computed Summary: {tabular_ctx.get('summary')}\n"
                f"Exact Result Data: {json.dumps(tabular_ctx.get('result', {}), default=str)}"
            )
            source_note = (
                "Answer based ONLY on the exact computed tabular results provided. "
                "State the specific numbers and entities clearly. "
                "Do NOT invent or extrapolate numerical figures. Cite the source as [Data: filename.csv]."
            )
        else:
            ctx_block = f"Tabular Query Error: {tabular_ctx.get('error')}"
            source_note = "Explain the error clearly to the user based on the error message above."
    elif context:
        ctx_block = "\n\n".join(
            f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
            for c in context
        )
        source_note = "Cite sources as [Source, Page X] in your answer."
    elif web_results:
        ctx_block   = "\n\n".join(web_results)
        source_note = "These results are from the web."
    else:
        ctx_block   = ""
        source_note = "Answer from your general knowledge."

    # Conversation history (last 6 turns for context window efficiency)
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history[-6:]
    )

    system = (
        "You are a precise, helpful document and data assistant. "
        "Answer only from the provided context. "
        "If the context is insufficient, say so clearly. "
        f"{source_note}"
    )
    user_prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Context:\n{ctx_block}\n\n"
        f"Question: {query}"
    )
    response = llm.invoke([
        HumanMessage(content=f"[SYSTEM]\n{system}\n\n[USER]\n{user_prompt}")
    ])
    answer = str(response.content)
    logger.info(f"[generate] answer length={len(answer)} chars")
    return {
        **state,
        "answer":   answer,
        "messages": state["messages"] + [AIMessage(content=answer)],
    }


# ── Conditional edges ─────────────────────────────────────────────────────────

def route_after_routing(state: AgentState) -> Literal["retrieve", "generate", "analyze_tabular"]:
    route = state.get("_route")
    if route == "documents":
        return "retrieve"
    elif route == "csv_data":
        return "analyze_tabular"
    return "generate"


def route_after_grading(state: AgentState) -> Literal["generate", "rewrite_query", "web_search"]:
    grade = state.get("_grade") or "generate"                               # type: ignore[union-attr]
    if grade == "generate":
        return "generate"
    # After MAX_REWRITES failed attempts, fall through to web search
    if int(state.get("rewrite_count") or 0) >= MAX_REWRITES:               # type: ignore[union-attr]
        return "web_search"
    return "rewrite_query"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("route_query",     route_query)
    g.add_node("retrieve",        retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query",   rewrite_query)
    g.add_node("web_search",      web_search)
    g.add_node("analyze_tabular", analyze_tabular)
    g.add_node("generate",        generate)

    g.set_entry_point("route_query")

    g.add_conditional_edges("route_query",     route_after_routing,
                            {"retrieve": "retrieve", "generate": "generate", "analyze_tabular": "analyze_tabular"})
    g.add_edge("retrieve",         "grade_documents")
    g.add_conditional_edges("grade_documents", route_after_grading,
                            {"generate": "generate",
                             "rewrite_query": "rewrite_query",
                             "web_search": "web_search"})
    g.add_edge("rewrite_query",    "retrieve")
    g.add_edge("web_search",       "generate")
    g.add_edge("analyze_tabular",  "generate")
    g.add_edge("generate",         END)

    return g.compile()


# Singleton compiled graph
rag_agent = build_graph()


# ── Public interface ──────────────────────────────────────────────────────────

def _to_messages(history: Optional[List[Any]]) -> List[BaseMessage]:
    """Ensure history items are LangChain BaseMessage objects."""
    if not history:
        return []
    converted: List[BaseMessage] = []
    for item in history:
        if isinstance(item, BaseMessage):
            converted.append(item)
        elif isinstance(item, dict):
            role = item.get("role", "").lower()
            content = str(item.get("content", ""))
            if role in ("user", "human"):
                converted.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                converted.append(AIMessage(content=content))
    return converted


def ask(query: str, history: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Main entry point. Call from FastAPI or Streamlit.
    Returns {"answer": str, "sources": list, "rewrite_count": int, "used_web": bool}
    """
    messages = _to_messages(history) + [HumanMessage(content=query)]
    initial_state: AgentState = {
        "messages":      messages,
        "query":         query,
        "rewrite_count": 0,
        "context":       [],
        "web_results":   [],
        "answer":        "",
    }
    final_state: Dict[str, Any] = rag_agent.invoke(initial_state)

    tabular_ctx = final_state.get("tabular_context")
    if tabular_ctx and tabular_ctx.get("status") == "success":
        sources = [{
            "text": tabular_ctx.get("summary", ""),
            "source": tabular_ctx.get("source", "dataset.csv"),
            "page": 0,
        }]
    else:
        sources = final_state.get("context", [])

    return {
        "answer":        final_state.get("answer", ""),
        "sources":       sources,
        "rewrite_count": final_state.get("rewrite_count", 0),
        "used_web":      bool(final_state.get("web_results")),
    }
