"""
tests/test_tabular_data.py
──────────────────────────
Self-contained unit test suite for Structured Data / CSV Analysis Tool.
Tests:
1. CSV loading, schema generation, and DataFrame creation.
2. Safe operations: highest, lowest, aggregation, frequency, filtering, groupby_agg.
3. Strict allowlist validation (operations, columns, operators, aggregations).
4. Error handling: missing CSV, missing column, non-numeric column.
5. LangGraph router selection:
   - csv_data route when tabular data query
   - documents route when document query
   - llm_only route when general knowledge query
6. Multi-turn contextualization for tabular queries.
7. Verification that existing single-turn RAG path remains intact.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide mock for fastapi and pydantic if not installed
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

    mock_pydantic = MagicMock()
    mock_pydantic.BaseModel = MockBaseModel

    sys.modules["fastapi"] = mock_fastapi
    sys.modules["fastapi.middleware"] = MagicMock()
    sys.modules["fastapi.middleware.cors"] = MagicMock()
    sys.modules["pydantic"] = mock_pydantic

# Mock external packages that may not be installed in the local environment
for mod in [
    "dotenv",
    "langchain_ollama",
    "langchain_community",
    "langchain_community.tools",
    "langchain_community.tools.tavily_search",
    "langgraph",
    "langgraph.graph",
    "langgraph.graph.message",
    "sentence_transformers",
    "faiss",
    "rank_bm25",
    "loguru"
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Provide real/mock message classes if langchain_core is not present
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

# Mock vectorstore before importing app modules
mock_store = MagicMock()
sys.modules["vectorstore"] = MagicMock()
sys.modules["vectorstore.store"] = mock_store
mock_ingest = MagicMock()
sys.modules["ingestion.ingest"] = mock_ingest

from ingestion.tabular_store import TabularStore, tabular_store
from app.agent import route_query, analyze_tabular, route_after_routing, generate


SAMPLE_CSV = """mission,agency,cost_millions,year,status
Apollo 11,NASA,355,1969,Success
Artemis 1,NASA,4100,2022,Success
Chandrayaan-3,ISRO,75,2023,Success
Luna 25,Roscosmos,130,2023,Failed
SLIM,JAXA,120,2024,Success
"""


class TestTabularDataAnalysis(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = TabularStore(data_dir=self.temp_dir.name)
        self.summary = self.store.load_csv(SAMPLE_CSV, "missions.csv")
        # Also populate singleton for agent tests
        tabular_store.data_dir = Path(self.temp_dir.name)
        tabular_store.load_csv(SAMPLE_CSV, "missions.csv")

    def tearDown(self):
        self.store.clear()
        tabular_store.clear()
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    # ── 1. CSV Loading & Ingestion ──────────────────────────────────────────

    def test_load_csv_valid(self):
        """Test loading valid CSV parses correct row count, columns, and types."""
        self.assertTrue(self.store.has_data())
        self.assertEqual(self.summary["row_count"], 5)
        self.assertIn("mission", self.summary["columns"])
        self.assertIn("cost_millions", self.summary["columns"])
        self.assertIn("year", self.summary["columns"])

    def test_load_csv_empty_or_invalid(self):
        """Test that empty or non-CSV files raise ValueError."""
        with self.assertRaises(ValueError):
            self.store.load_csv("", "empty.csv")
        with self.assertRaises(ValueError):
            self.store.load_csv("hello", "document.pdf")

    def test_schema_summary(self):
        """Test schema description contains column names and types."""
        schema = self.store.get_schema_summary()
        self.assertIn("missions.csv", schema)
        self.assertIn("cost_millions", schema)
        self.assertIn("numeric", schema)

    # ── 2. Safe Operations & Calculations ───────────────────────────────────

    def test_op_highest(self):
        """Test highest value query: Artemis 1 should have highest cost ($4100M)."""
        result = self.store.execute_query({
            "operation": "highest",
            "target_column": "cost_millions"
        })
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["mission"], "Artemis 1")
        self.assertEqual(result["result"]["cost_millions"], 4100)
        self.assertIn("4100", result["summary"])

    def test_op_lowest(self):
        """Test lowest value query: Chandrayaan-3 should have lowest cost ($75M)."""
        result = self.store.execute_query({
            "operation": "lowest",
            "target_column": "cost_millions"
        })
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["mission"], "Chandrayaan-3")
        self.assertEqual(result["result"]["cost_millions"], 75)

    def test_op_aggregate(self):
        """Test statistical aggregations: sum, mean, count, min, max."""
        sum_res = self.store.execute_query({
            "operation": "aggregate",
            "target_column": "cost_millions",
            "agg_func": "sum"
        })
        self.assertEqual(sum_res["status"], "success")
        self.assertEqual(sum_res["result"]["cost_millions"], 4780)

        mean_res = self.store.execute_query({
            "operation": "aggregate",
            "target_column": "cost_millions",
            "agg_func": "mean"
        })
        self.assertEqual(mean_res["status"], "success")
        self.assertEqual(mean_res["result"]["cost_millions"], 956.0)

        count_res = self.store.execute_query({
            "operation": "aggregate",
            "target_column": "mission",
            "agg_func": "count"
        })
        self.assertEqual(count_res["status"], "success")
        self.assertEqual(count_res["result"]["mission"], 5)

    def test_op_frequency(self):
        """Test frequency query: NASA is the most frequent agency (2 missions)."""
        res = self.store.execute_query({
            "operation": "frequency",
            "target_column": "agency"
        })
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["result"]["most_frequent"], "NASA")
        self.assertEqual(res["result"]["count"], 2)

    def test_op_filter(self):
        """Test safe filtering with allowed operators (==, >, contains)."""
        # Equality filter
        res_eq = self.store.execute_query({
            "operation": "filter",
            "filter": {"column": "agency", "operator": "==", "value": "NASA"}
        })
        self.assertEqual(res_eq["status"], "success")
        self.assertEqual(res_eq["row_count"], 2)

        # Numeric greater-than filter
        res_gt = self.store.execute_query({
            "operation": "filter",
            "filter": {"column": "cost_millions", "operator": ">", "value": 200}
        })
        self.assertEqual(res_gt["status"], "success")
        self.assertEqual(res_gt["row_count"], 2)

        # Substring contains filter
        res_contains = self.store.execute_query({
            "operation": "filter",
            "filter": {"column": "mission", "operator": "contains", "value": "Apollo"}
        })
        self.assertEqual(res_contains["status"], "success")
        self.assertEqual(res_contains["row_count"], 1)

    def test_op_groupby_agg(self):
        """Test groupby and aggregation: total cost per agency."""
        res = self.store.execute_query({
            "operation": "groupby_agg",
            "group_column": "agency",
            "target_column": "cost_millions",
            "agg_func": "sum"
        })
        self.assertEqual(res["status"], "success")
        # NASA: 355 + 4100 = 4455
        self.assertEqual(res["result"]["NASA"], 4455)
        # ISRO: 75
        self.assertEqual(res["result"]["ISRO"], 75)

    # ── 3. Strict Allowlists & Security Validation ──────────────────────────

    def test_disallowed_operation(self):
        """Test that disallowed operations (e.g. exec, eval, drop) are rejected."""
        res = self.store.execute_query({"operation": "eval", "target_column": "mission"})
        self.assertEqual(res["status"], "error")
        self.assertIn("Invalid operation", res["error"])

    def test_disallowed_filter_operator(self):
        """Test that unapproved filter operators (e.g. SQL injection attempts) are rejected."""
        res = self.store.execute_query({
            "operation": "filter",
            "filter": {"column": "agency", "operator": "; DROP TABLE", "value": "NASA"}
        })
        self.assertEqual(res["status"], "error")
        self.assertIn("not allowed", res["error"])

    def test_missing_column_error(self):
        """Test that non-existent column returns clean error listing available columns."""
        res = self.store.execute_query({
            "operation": "highest",
            "target_column": "non_existent_column"
        })
        self.assertEqual(res["status"], "error")
        self.assertIn("not found", res["error"])
        self.assertIn("cost_millions", res["error"])

    def test_non_numeric_aggregation_error(self):
        """Test that computing sum/mean on text columns is prevented."""
        res = self.store.execute_query({
            "operation": "aggregate",
            "target_column": "agency",
            "agg_func": "mean"
        })
        self.assertEqual(res["status"], "error")
        self.assertIn("Cannot compute", res["error"])

    def test_missing_csv_handling(self):
        """Test error handling when no CSV is loaded."""
        empty_store = TabularStore()
        res = empty_store.execute_query({"operation": "highest", "target_column": "cost_millions"})
        self.assertEqual(res["status"], "error")
        self.assertIn("No CSV dataset is currently loaded", res["error"])

    # ── 4. LangGraph Router Tests ───────────────────────────────────────────

    def test_router_selects_csv_data_for_tabular_query(self):
        """Test router selects 'csv_data' when active CSV is loaded and query asks for numbers."""
        state = {
            "messages": [HumanMessage(content="Which mission had the highest cost?")],
            "query": "Which mission had the highest cost?",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
        }

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "csv_data"
        mock_llm.invoke.return_value = mock_resp

        with patch("app.agent.llm", mock_llm):
            result = route_query(state)

        self.assertEqual(result["_route"], "csv_data")
        # Route after routing should direct to analyze_tabular
        self.assertEqual(route_after_routing(result), "analyze_tabular")

    def test_router_selects_documents_for_doc_query(self):
        """Test router selects 'documents' for text document queries even if CSV is loaded."""
        state = {
            "messages": [HumanMessage(content="What does the PDF document say about security?")],
            "query": "What does the PDF document say about security?",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
        }

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "documents"
        mock_llm.invoke.return_value = mock_resp

        with patch("app.agent.llm", mock_llm):
            result = route_query(state)

        self.assertEqual(result["_route"], "documents")
        self.assertEqual(route_after_routing(result), "retrieve")

    def test_router_selects_llm_only_for_general_query(self):
        """Test router selects 'llm_only' for general knowledge."""
        state = {
            "messages": [HumanMessage(content="What is the capital of France?")],
            "query": "What is the capital of France?",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
        }

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "llm_only"
        mock_llm.invoke.return_value = mock_resp

        with patch("app.agent.llm", mock_llm):
            result = route_query(state)

        self.assertEqual(result["_route"], "llm_only")
        self.assertEqual(route_after_routing(result), "generate")

    # ── 5. Multi-Turn Tabular Question Contextualization ────────────────────

    def test_multiturn_tabular_coreference(self):
        """
        Test multi-turn dialog referencing earlier entity:
        Turn 1: "Show me the missions in the dataset."
        Turn 2: "Which one had the highest cost?"
        """
        history_messages = [
            HumanMessage(content="Show me the missions in the dataset."),
            AIMessage(content="The missions are Apollo 11, Artemis 1, Chandrayaan-3, Luna 25, and SLIM."),
            HumanMessage(content="Which one had the highest cost?")
        ]
        state = {
            "messages": history_messages,
            "query": "Which one had the highest cost?",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
        }

        mock_llm = MagicMock()
        # Call 1: route -> csv_data
        # Call 2: contextualize -> "Which mission in the dataset had the highest cost?"
        resp1 = MagicMock()
        resp1.content = "csv_data"
        resp2 = MagicMock()
        resp2.content = "Which mission in the dataset had the highest cost?"
        mock_llm.invoke.side_effect = [resp1, resp2]

        with patch("app.agent.llm", mock_llm):
            result = route_query(state)

        self.assertEqual(result["_route"], "csv_data")
        self.assertEqual(result["query"], "Which mission in the dataset had the highest cost?")

    # ── 6. Node Execution & Answer Grounding ────────────────────────────────

    def test_analyze_tabular_node_executes_safely(self):
        """Test analyze_tabular node calls LLM for parameter plan and executes against tabular_store."""
        state = {
            "messages": [HumanMessage(content="Which mission cost the most?")],
            "query": "Which mission cost the most?",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
        }

        mock_llm = MagicMock()
        mock_plan_resp = MagicMock()
        mock_plan_resp.content = '{"operation": "highest", "target_column": "cost_millions"}'
        mock_llm.invoke.return_value = mock_plan_resp

        with patch("app.agent.llm", mock_llm):
            result_state = analyze_tabular(state)

        self.assertIn("tabular_context", result_state)
        ctx = result_state["tabular_context"]
        self.assertEqual(ctx["status"], "success")
        self.assertEqual(ctx["result"]["mission"], "Artemis 1")
        self.assertEqual(ctx["result"]["cost_millions"], 4100)

    def test_generate_grounds_on_exact_tabular_result(self):
        """Test generate node incorporates computed tabular results into prompt with source note."""
        state = {
            "messages": [HumanMessage(content="Which mission had the highest cost?")],
            "query": "Which mission had the highest cost?",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "tabular_context": {
                "status": "success",
                "source": "missions.csv",
                "operation": "highest",
                "summary": "The highest cost_millions is 4100 (Artemis 1).",
                "result": {"mission": "Artemis 1", "cost_millions": 4100}
            },
            "answer": "",
        }

        mock_llm = MagicMock()
        mock_ans = MagicMock()
        mock_ans.content = "Artemis 1 had the highest cost at $4,100 million. [Data: missions.csv]"
        mock_llm.invoke.return_value = mock_ans

        with patch("app.agent.llm", mock_llm):
            final_state = generate(state)

        self.assertEqual(final_state["answer"], "Artemis 1 had the highest cost at $4,100 million. [Data: missions.csv]")
        # Verify the prompt sent to LLM contains the exact computed data
        prompt_content = mock_llm.invoke.call_args[0][0][0].content
        self.assertIn("Artemis 1", prompt_content)
        self.assertIn("4100", prompt_content)
        self.assertIn("Answer based ONLY on the exact computed tabular results provided", prompt_content)


if __name__ == "__main__":
    unittest.main()
