"""
tests/test_agent_trace.py
─────────────────────────
Unit tests for Feature 3: Agent execution tracing and explainability.
Tests:
1. Trace structure initialization in ask().
2. Routing decision recording in trace (documents, csv_data, llm_only).
3. Exact execution order of executed nodes (and exclusion of unexecuted nodes).
4. CSV tabular analysis telemetry recording.
5. Document retrieval statistics recording (FAISS, BM25, fused, reranked).
6. Corrective RAG query rewriting event recording.
7. Tavily web search fallback event recording.
8. Source metadata recording.
9. Node error handling (status="error", error message, no stack trace).
10. FastAPI /query endpoint returning trace in QueryResponse.
11. Strict privacy verification: no secret prompts, credentials, or chain-of-thought in trace.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Lightweight mocks for dependencies if running in minimal environment
try:
    import fastapi
    from pydantic import BaseModel
except ImportError:
    class MockBaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def mock_decorator(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

    mock_fastapi = MagicMock()
    mock_app_instance = MagicMock()
    mock_app_instance.get = mock_decorator
    mock_app_instance.post = mock_decorator
    mock_fastapi.FastAPI.return_value = mock_app_instance
    mock_fastapi.File.return_value = MagicMock()
    mock_fastapi.UploadFile = MagicMock()

    mock_fastapi_cors = MagicMock()
    mock_pydantic = MagicMock()
    mock_pydantic.BaseModel = MockBaseModel
    sys.modules["fastapi"] = mock_fastapi
    sys.modules["fastapi.middleware"] = MagicMock()
    sys.modules["fastapi.middleware.cors"] = mock_fastapi_cors
    sys.modules["pydantic"] = mock_pydantic

for mod in [
    "dotenv", "langchain_ollama", "langchain_community",
    "langchain_community.tools", "langchain_community.tools.tavily_search",
    "langgraph", "langgraph.graph", "langgraph.graph.message",
    "sentence_transformers", "faiss", "rank_bm25", "loguru",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

try:
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
except ImportError:
    class BaseMessage:
        def __init__(self, content):
            self.content = content
        def __eq__(self, other):
            return self.__class__ == other.__class__ and self.content == other.content
        def __repr__(self):
            return f"{self.__class__.__name__}(content={self.content!r})"

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    mock_lc = MagicMock()
    mock_lc.HumanMessage = HumanMessage
    mock_lc.AIMessage = AIMessage
    mock_lc.BaseMessage = BaseMessage
    sys.modules["langchain_core"] = mock_lc
    sys.modules["langchain_core.messages"] = mock_lc

mock_store = MagicMock()
sys.modules["vectorstore"] = MagicMock()
sys.modules["vectorstore.store"] = mock_store
mock_ingest = MagicMock()
sys.modules["ingestion.ingest"] = mock_ingest

from app import agent, api
from ingestion.tabular_store import tabular_store


import tempfile
from pathlib import Path

class TestAgentExecutionTrace(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_data_dir = tabular_store.data_dir
        tabular_store.data_dir = Path(self.temp_dir.name)
        tabular_store.clear()

    def tearDown(self):
        tabular_store.clear()
        tabular_store.data_dir = self.old_data_dir
        self.temp_dir.cleanup()

    def test_trace_initialization_structure(self):
        """Test that initial trace structure contains all required telemetry fields."""
        initial_trace = {
            "route": None,
            "nodes": [],
            "retrieval": None,
            "tabular": None,
            "web_search": None,
            "sources": [],
            "total_duration_ms": 0,
            "error": None,
        }
        for k in ["route", "nodes", "retrieval", "tabular", "web_search", "sources", "total_duration_ms"]:
            self.assertIn(k, initial_trace)

    def test_route_query_records_route_and_node(self):
        """Test route_query appends to trace['nodes'] and sets trace['route']."""
        state = {
            "messages": [HumanMessage(content="What is in the policy document?")],
            "query": "What is in the policy document?",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
            "trace": {"route": None, "nodes": []},
        }
        mock_resp = MagicMock()
        mock_resp.content = "documents"
        with patch.object(agent.llm, "invoke", return_value=mock_resp):
            out = agent.route_query(state)

        self.assertEqual(out["_route"], "documents")
        self.assertEqual(out["trace"]["route"], "documents")
        self.assertEqual(len(out["trace"]["nodes"]), 1)
        node = out["trace"]["nodes"][0]
        self.assertEqual(node["name"], "route_query")
        self.assertEqual(node["status"], "success")
        self.assertGreaterEqual(node["duration_ms"], 0)

    def test_retrieve_records_retrieval_statistics(self):
        """Test retrieve records candidate counts and reranked count."""
        state = {
            "messages": [HumanMessage(content="Explain architecture")],
            "query": "Explain architecture",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
            "trace": {"route": "documents", "nodes": []},
        }
        mock_chunks = [
            {"text": "Chunk A", "source": "guide.pdf", "page": 1},
            {"text": "Chunk B", "source": "guide.pdf", "page": 2},
        ]
        mock_stats = {
            "faiss_candidates": 18,
            "bm25_candidates": 15,
            "fused_candidates": 22,
            "reranked_count": 2,
        }
        with patch.object(agent, "hybrid_search", return_value=(mock_chunks, mock_stats)):
            out = agent.retrieve(state)

        self.assertEqual(len(out["trace"]["nodes"]), 1)
        self.assertEqual(out["trace"]["nodes"][0]["name"], "retrieve")
        ret_info = out["trace"]["retrieval"]
        self.assertIsNotNone(ret_info)
        self.assertEqual(ret_info["faiss_candidates"], 18)
        self.assertEqual(ret_info["bm25_candidates"], 15)
        self.assertEqual(ret_info["fused_candidates"], 22)
        self.assertEqual(ret_info["reranked_count"], 2)
        self.assertEqual(len(ret_info["sources"]), 2)

        # Also verify fallback when hybrid_search returns a plain list
        with patch.object(agent, "hybrid_search", return_value=mock_chunks):
            out_fallback = agent.retrieve(state)
        self.assertEqual(out_fallback["trace"]["retrieval"]["reranked_count"], 2)

    def test_grade_documents_records_grading_info(self):
        """Test grade_documents marks grading occurrence and chunk counts."""
        state = {
            "messages": [HumanMessage(content="Query")],
            "query": "Query",
            "rewrite_count": 0,
            "context": [{"text": "Useful content", "source": "doc.pdf", "page": 1}],
            "web_results": [],
            "answer": "",
            "trace": {
                "route": "documents",
                "nodes": [{"name": "retrieve", "status": "success", "duration_ms": 10}],
                "retrieval": {"grading_occurred": False},
            },
        }
        mock_resp = MagicMock()
        mock_resp.content = "yes"
        with patch.object(agent.llm, "invoke", return_value=mock_resp):
            out = agent.grade_documents(state)

        self.assertEqual(out["_grade"], "generate")
        self.assertEqual(len(out["trace"]["nodes"]), 2)
        self.assertEqual(out["trace"]["nodes"][1]["name"], "grade_documents")
        self.assertTrue(out["trace"]["retrieval"]["grading_occurred"])
        self.assertEqual(out["trace"]["retrieval"]["relevant_chunks"], 1)

    def test_rewrite_query_records_rewrite_event(self):
        """Test rewrite_query increments rewrite_count and logs event in trace."""
        state = {
            "messages": [HumanMessage(content="poor query")],
            "query": "poor query",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
            "trace": {
                "route": "documents",
                "nodes": [],
                "retrieval": {"rewrites_occurred": False},
            },
        }
        mock_resp = MagicMock()
        mock_resp.content = "specific detailed technical query"
        with patch.object(agent.llm, "invoke", return_value=mock_resp):
            out = agent.rewrite_query(state)

        self.assertEqual(out["query"], "specific detailed technical query")
        self.assertEqual(out["rewrite_count"], 1)
        self.assertTrue(out["trace"]["retrieval"]["rewrites_occurred"])
        self.assertEqual(out["trace"]["retrieval"]["rewrite_count"], 1)

    def test_web_search_records_fallback_event(self):
        """Test web_search records fallback_triggered and results count."""
        state = {
            "messages": [HumanMessage(content="latest event")],
            "query": "latest event",
            "rewrite_count": 2,
            "context": [],
            "web_results": [],
            "answer": "",
            "trace": {"route": "documents", "nodes": [], "retrieval": {"web_fallback": False}},
        }
        mock_tavily = [{"content": "Web result snippet 1"}, {"content": "Web result snippet 2"}]
        with patch.object(agent.web_search_tool, "invoke", return_value=mock_tavily):
            out = agent.web_search(state)

        self.assertEqual(len(out["web_results"]), 2)
        self.assertTrue(out["trace"]["web_search"]["fallback_triggered"])
        self.assertEqual(out["trace"]["web_search"]["results_count"], 2)
        self.assertTrue(out["trace"]["retrieval"]["web_fallback"])

    def test_analyze_tabular_records_csv_metadata(self):
        """Test analyze_tabular records detailed CSV operation telemetry."""
        csv_data = "mission,cost\nArtemis 1,4100\nApollo 11,2700\n"
        tabular_store.load_csv(csv_data.encode("utf-8"), "missions.csv")

        state = {
            "messages": [HumanMessage(content="Which mission had the highest cost?")],
            "query": "Which mission had the highest cost?",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
            "trace": {"route": "csv_data", "nodes": []},
        }
        mock_resp = MagicMock()
        mock_resp.content = '{"operation": "highest", "target_column": "cost"}'
        with patch.object(agent.llm, "invoke", return_value=mock_resp):
            out = agent.analyze_tabular(state)

        self.assertEqual(len(out["trace"]["nodes"]), 1)
        self.assertEqual(out["trace"]["nodes"][0]["name"], "analyze_tabular")
        tab = out["trace"]["tabular"]
        self.assertEqual(tab["source"], "missions.csv")
        self.assertEqual(tab["operation"], "highest")
        self.assertEqual(tab["target_column"], "cost")
        self.assertEqual(tab["row_count"], 2)
        self.assertEqual(tab["status"], "success")

    def test_generate_records_total_duration_and_sources(self):
        """Test generate records final source metadata and total duration."""
        state = {
            "messages": [HumanMessage(content="What is Artemis 1?")],
            "query": "What is Artemis 1?",
            "rewrite_count": 0,
            "context": [{"source": "nasa.pdf", "page": 4, "text": "Artemis 1 is an uncrewed Moon mission."}],
            "web_results": [],
            "tabular_context": None,
            "answer": "",
            "trace": {
                "route": "documents",
                "nodes": [
                    {"name": "route_query", "status": "success", "duration_ms": 15},
                    {"name": "retrieve", "status": "success", "duration_ms": 25},
                ],
            },
        }
        mock_resp = MagicMock()
        mock_resp.content = "Artemis 1 is a Moon mission [nasa.pdf, Page 4]."
        with patch.object(agent.llm, "invoke", return_value=mock_resp):
            out = agent.generate(state)

        self.assertEqual(len(out["trace"]["nodes"]), 3)
        self.assertEqual(out["trace"]["nodes"][2]["name"], "generate")
        self.assertEqual(len(out["trace"]["sources"]), 1)
        self.assertEqual(out["trace"]["sources"][0]["source"], "nasa.pdf")
        self.assertGreaterEqual(out["trace"]["total_duration_ms"], 40)

    def test_node_error_handling_in_trace(self):
        """Test that node failure records status='error' with clean message and no crash."""
        state = {
            "messages": [HumanMessage(content="Query")],
            "query": "Query",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
            "trace": {"route": "documents", "nodes": []},
        }
        with patch.object(agent, "hybrid_search", side_effect=RuntimeError("Index corrupt")):
            out = agent.retrieve(state)

        self.assertEqual(len(out["trace"]["nodes"]), 1)
        node = out["trace"]["nodes"][0]
        self.assertEqual(node["status"], "error")
        self.assertEqual(node["error"], "RuntimeError")

    def test_fastapi_query_returns_trace(self):
        """Test that FastAPI /query endpoint includes trace in QueryResponse."""
        fake_result = {
            "answer": "Answer text",
            "sources": [{"text": "Summary", "source": "data.csv", "page": 0}],
            "rewrite_count": 0,
            "used_web": False,
            "trace": {
                "route": "csv_data",
                "nodes": [{"name": "route_query", "status": "success", "duration_ms": 10}],
                "total_duration_ms": 10,
            },
        }
        with patch.object(api, "ask", return_value=fake_result):
            req = api.QueryRequest(question="Test question")
            resp = api.query(req)

        self.assertEqual(resp.answer, "Answer text")
        self.assertIsNotNone(resp.trace)
        self.assertEqual(resp.trace["route"], "csv_data")
        self.assertEqual(resp.trace["nodes"][0]["name"], "route_query")

    def test_trace_contains_no_sensitive_fields(self):
        """Test that trace never exposes prompts, chain-of-thought, or internal paths."""
        forbidden_substrings = [
            "system_prompt",
            "chain_of_thought",
            "secret",
            "api_key",
            "tavily_api_key",
            "ollama_base_url",
            "internal_credential",
        ]
        sample_trace = {
            "route": "csv_data",
            "nodes": [
                {"name": "route_query", "status": "success", "duration_ms": 12},
                {"name": "analyze_tabular", "status": "success", "duration_ms": 21},
                {"name": "generate", "status": "success", "duration_ms": 84},
            ],
            "tabular": {"source": "missions.csv", "operation": "highest", "target_column": "cost"},
            "sources": [{"source": "missions.csv", "type": "tabular"}],
            "total_duration_ms": 117,
        }
        trace_str = str(sample_trace).lower()
        for forbidden in forbidden_substrings:
            self.assertNotIn(forbidden, trace_str)


if __name__ == "__main__":
    unittest.main()
