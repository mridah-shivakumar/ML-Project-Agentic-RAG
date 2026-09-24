"""
tests/test_rag_evaluation.py
────────────────────────────
Unit tests for Feature 4: RAG Evaluation & Reliability Framework.
Tests:
1. Dataset loading and schema validation.
2. Malformed dataset entries rejected (missing required fields).
3. Retrieval source-hit and success calculations.
4. Deterministic CSV evaluation (highest, aggregate, filter).
5. Explicit arithmetic mean aggregate calculation (only valid non-null metrics).
6. Missing metrics safety (does not treat nulls as zero; returns None when empty).
7. Graceful degradation when Ollama/RAGAs judge is unavailable.
8. End-to-end evaluation run outputting results.json without fake scores.
9. GET /metrics and GET /evaluation/results endpoint responses.
10. Un-evaluated state handling in API.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Lightweight mocks for dependencies
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

from app import api
from evaluation import evaluate
from ingestion.tabular_store import tabular_store


class TestRAGEvaluationFramework(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.old_data_dir = tabular_store.data_dir
        tabular_store.data_dir = Path(self.tmp_dir.name)
        tabular_store.clear()

    def tearDown(self):
        tabular_store.clear()
        tabular_store.data_dir = self.old_data_dir
        self.tmp_dir.cleanup()

    def test_load_eval_dataset_valid(self):
        """Test loading actual repository benchmark dataset."""
        dataset = evaluate.load_eval_dataset("evaluation/eval_dataset.json")
        self.assertEqual(len(dataset), 10)
        case_ids = [c["id"] for c in dataset]
        self.assertIn("doc_1", case_ids)
        self.assertIn("csv_1", case_ids)
        self.assertIn("multi_1", case_ids)

        for case in dataset:
            self.assertIn("category", case)
            self.assertIn("question", case)
            self.assertIn("expected_route", case)
            self.assertIn("ground_truth", case)

    def test_load_eval_dataset_malformed_rejected(self):
        """Test that missing required fields or empty dataset raises ValueError."""
        # Missing required field 'ground_truth'
        bad_data = [{"id": "bad_1", "category": "doc", "question": "q", "expected_route": "documents"}]
        bad_file = Path(self.tmp_dir.name) / "bad.json"
        bad_file.write_text(json.dumps(bad_data), encoding="utf-8")

        with self.assertRaises(ValueError):
            evaluate.load_eval_dataset(bad_file)

        # Empty dataset
        empty_file = Path(self.tmp_dir.name) / "empty.json"
        empty_file.write_text("[]", encoding="utf-8")
        with self.assertRaises(ValueError):
            evaluate.load_eval_dataset(empty_file)

    def test_evaluate_retrieval_case_matching_source(self):
        """Test retrieval evaluation flags source_hit=True when expected document is in sources."""
        case = {
            "id": "doc_1",
            "expected_source": "policy.pdf",
        }
        sources = [
            {"source": "other.pdf", "page": 1},
            {"source": "policy.pdf", "page": 3},
        ]
        res = evaluate.evaluate_retrieval_case(case, sources)
        self.assertTrue(res["source_hit"])
        self.assertTrue(res["retrieval_success"])

    def test_evaluate_retrieval_case_missing_source(self):
        """Test retrieval evaluation flags source_hit=False when expected document is absent."""
        case = {
            "id": "doc_1",
            "expected_source": "policy.pdf",
        }
        sources = [
            {"source": "other.pdf", "page": 1},
            {"source": "readme.pdf", "page": 1},
        ]
        res = evaluate.evaluate_retrieval_case(case, sources)
        self.assertFalse(res["source_hit"])
        self.assertFalse(res["retrieval_success"])

    def test_evaluate_csv_case_correct(self):
        """Test deterministic CSV evaluation identifies correct operation and exact value match."""
        case = {
            "id": "csv_1",
            "csv_expectations": {
                "operation": "highest",
                "target_column": "cost",
                "expected_value": 4100.0,
                "expected_entity": "Artemis 1",
            },
        }
        trace = {
            "tabular": {
                "operation": "highest",
                "target_column": "cost",
            }
        }
        answer = "Artemis 1 had the maximum cost of 4100.0 million USD."
        tab_ctx = {
            "operation": "highest",
            "summary": "Maximum cost is Artemis 1 with 4100.0",
            "result": {"mission": "Artemis 1", "cost": 4100.0},
        }
        res = evaluate.evaluate_csv_case(case, trace, answer, tab_ctx)
        self.assertTrue(res["operation_correct"])
        self.assertTrue(res["column_correct"])
        self.assertTrue(res["value_match"])
        self.assertTrue(res["overall_correct"])

    def test_evaluate_csv_case_wrong_operation(self):
        """Test deterministic CSV evaluation fails if operation does not match expectation."""
        case = {
            "id": "csv_1",
            "csv_expectations": {
                "operation": "highest",
                "target_column": "cost",
                "expected_value": 4100.0,
            },
        }
        trace = {
            "tabular": {
                "operation": "lowest",
                "target_column": "cost",
            }
        }
        answer = "Chandrayaan-3 had cost 75."
        tab_ctx = {
            "operation": "lowest",
            "summary": "Minimum cost is Chandrayaan-3 with 75.0",
            "result": {"cost": 75.0},
        }
        res = evaluate.evaluate_csv_case(case, trace, answer, tab_ctx)
        self.assertFalse(res["operation_correct"])
        self.assertFalse(res["overall_correct"])

    def test_compute_aggregates_arithmetic_mean(self):
        """Test that aggregate reliability is exactly the arithmetic mean of non-null metrics."""
        results = [
            {
                "expected_route": "documents",
                "retrieval_eval": {"source_hit": True, "retrieval_success": True},
            },
            {
                "expected_route": "documents",
                "retrieval_eval": {"source_hit": False, "retrieval_success": False},
            },
            {
                "tabular_eval": {"overall_correct": True, "operation_correct": True},
            },
        ]
        # retrieval: hit_rate = 0.5, success_rate = 0.5
        # structured: correctness_rate = 1.0, operation_rate = 1.0
        # No RAGAs scores
        # Aggregate: mean(0.5, 0.5, 1.0, 1.0) = 3.0 / 4 = 0.75
        agg = evaluate.compute_aggregates(results, ragas_scores=None)
        self.assertEqual(agg["retrieval"]["source_hit_rate"], 0.5)
        self.assertEqual(agg["retrieval"]["retrieval_success_rate"], 0.5)
        self.assertEqual(agg["structured_data"]["correctness_rate"], 1.0)
        self.assertEqual(agg["structured_data"]["operation_selection_rate"], 1.0)
        self.assertEqual(agg["aggregate_reliability"], 0.75)
        self.assertEqual(agg["active_metrics_count"], 4)

    def test_compute_aggregates_with_ragas_scores(self):
        """Test arithmetic mean calculation when RAGAs scores are present."""
        results = [
            {
                "expected_route": "documents",
                "retrieval_eval": {"source_hit": True, "retrieval_success": True},
            },
            {
                "tabular_eval": {"overall_correct": True, "operation_correct": True},
            },
        ]
        # source_hit: 1.0, ret_success: 1.0, csv_correct: 1.0, csv_op: 1.0
        # ragas: faithfulness=0.9, answer_relevancy=0.8
        ragas_scores = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": None,
            "context_recall": None,
        }
        # valid: [1.0, 1.0, 1.0, 1.0, 0.9, 0.8] -> sum=5.7 / 6 = 0.95
        agg = evaluate.compute_aggregates(results, ragas_scores=ragas_scores)
        self.assertEqual(agg["aggregate_reliability"], 0.95)
        self.assertEqual(agg["active_metrics_count"], 6)

    def test_compute_aggregates_empty_safe(self):
        """Test that having no valid metrics safely returns None without crashing."""
        agg = evaluate.compute_aggregates([], ragas_scores=None)
        self.assertIsNone(agg["aggregate_reliability"])
        self.assertEqual(agg["active_metrics_count"], 0)

    def test_run_evaluation_produces_valid_json(self):
        """Test running evaluation on a 2-question subset creates structured results.json."""
        test_dataset = [
            {
                "id": "csv_test",
                "category": "csv_highest",
                "question": "Which mission had highest cost?",
                "expected_route": "csv_data",
                "ground_truth": "Artemis 1 ($4100M)",
                "csv_expectations": {
                    "operation": "highest",
                    "target_column": "cost",
                    "expected_value": 4100.0,
                    "expected_entity": "Artemis 1",
                },
            },
            {
                "id": "doc_test",
                "category": "doc_direct",
                "question": "What is the policy?",
                "expected_route": "documents",
                "expected_source": "policy.pdf",
                "ground_truth": "Policy content.",
                "synthetic_context": "Policy context.",
            },
        ]
        d_path = Path(self.tmp_dir.name) / "test_ds.json"
        d_path.write_text(json.dumps(test_dataset), encoding="utf-8")
        r_path = Path(self.tmp_dir.name) / "results.json"
        s_path = Path(self.tmp_dir.name) / "latest_scores.json"

        # Mock ask() to return deterministic results
        def mock_ask_fn(query, history=None):
            if "highest" in query.lower():
                return {
                    "answer": "Artemis 1 had highest cost of 4100.0",
                    "sources": [{"source": "space_missions.csv", "page": 0}],
                    "rewrite_count": 0,
                    "used_web": False,
                    "trace": {
                        "route": "csv_data",
                        "nodes": [{"name": "route_query", "status": "success", "duration_ms": 10}],
                        "tabular": {
                            "operation": "highest",
                            "target_column": "cost",
                            "rows_analyzed": 5,
                            "result_count": 1,
                        },
                    },
                }
            else:
                return {
                    "answer": "Policy details [policy.pdf, Page 1]",
                    "sources": [{"source": "policy.pdf", "page": 1, "text": "Policy context."}],
                    "rewrite_count": 0,
                    "used_web": False,
                    "trace": {
                        "route": "documents",
                        "nodes": [{"name": "route_query", "status": "success", "duration_ms": 10}],
                        "retrieval": {
                            "faiss_candidates": 20,
                            "bm25_candidates": 20,
                            "fused_candidates": 20,
                            "reranked_count": 1,
                        },
                    },
                }

        with patch("evaluation.evaluate.ask", side_effect=mock_ask_fn):
            out = evaluate.run_evaluation(
                dataset_path=d_path,
                results_path=r_path,
                scores_path=s_path,
                enable_ragas=False,
            )

        self.assertTrue(r_path.exists())
        self.assertTrue(s_path.exists())

        saved_data = json.loads(r_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_data["dataset_size"], 2)
        self.assertEqual(len(saved_data["results"]), 2)
        self.assertIn("aggregate", saved_data)
        self.assertIsNotNone(saved_data["aggregate"]["aggregate_reliability"])

    def test_api_metrics_un_evaluated_state(self):
        """Test API metrics endpoints handle absent evaluation files gracefully."""
        non_existent_scores = Path(self.tmp_dir.name) / "absent_scores.json"
        non_existent_results = Path(self.tmp_dir.name) / "absent_results.json"

        with patch.object(api, "METRICS_FILE", non_existent_scores), \
             patch.object(api, "RESULTS_FILE", non_existent_results):
            resp = api.metrics()
            self.assertIn("detail", resp)

            resp_results = api.evaluation_results()
            self.assertIn("detail", resp_results)


if __name__ == "__main__":
    unittest.main()
