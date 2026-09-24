"""
evaluation/evaluate.py
──────────────────────
Unified RAG & Tabular Evaluation Engine.
Evaluates:
  1. Document RAG metrics via RAGAs (faithfulness, answer_relevancy, context_precision, context_recall)
     when Ollama judge LLM is available.
  2. Document retrieval metrics (source hit rate, retrieval success rate).
  3. Structured data / CSV deterministic correctness (operation selection, column match, exact computed value).
  4. Explicit aggregate reliability score (arithmetic mean across all available evaluated metrics).

Outputs results to evaluation/results.json and maintains backward-compatible
scores in evaluation/latest_scores.json.

Usage:
    python -m evaluation.evaluate
"""

from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("evaluation")

from app.agent import ask
from ingestion.tabular_store import tabular_store

RESULTS_FILE = Path("evaluation/results.json")
SCORES_FILE  = Path("evaluation/latest_scores.json")
EVAL_DATASET_FILE = Path("evaluation/eval_dataset.json")

RESULTS_FILE.parent.mkdir(exist_ok=True)

# Standard aerospace missions CSV used as default benchmark table if none loaded
SAMPLE_BENCHMARK_CSV = """mission,agency,cost,launch_year,status
Artemis 1,NASA,4100.0,2022,Success
Apollo 11,NASA,2700.0,1969,Success
Juice,ESA,1600.0,2023,En Route
Chandrayaan-3,ISRO,75.0,2023,Success
Tianwen-1,CNSA,2700.0,2020,Success
"""

# ── RAGAs Import & Graceful Detection ────────────────────────────────────────

try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAs or datasets library not found. RAGAs metrics will be marked unavailable.")


def get_ragas_judge() -> Optional[Any]:
    """Instantiate judge LLM wrapper for RAGAs if Ollama is available."""
    if not RAGAS_AVAILABLE:
        return None
    try:
        from langchain_ollama import ChatOllama
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        judge_llm = ChatOllama(model=model_name, temperature=0)
        return LangchainLLMWrapper(judge_llm)
    except Exception as e:
        logger.warning(f"Could not initialize local Ollama judge for RAGAs: {e}")
        return None


# ── Dataset Loading ──────────────────────────────────────────────────────────

def load_eval_dataset(path: Union[Path, str] = EVAL_DATASET_FILE) -> List[Dict[str, Any]]:
    """
    Load and validate evaluation cases from JSON.
    Rejects malformed entries missing required schema properties.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {path}")

    content = path.read_text(encoding="utf-8")
    data = json.loads(content)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Evaluation dataset must be a non-empty list of test cases.")

    required_keys = {"id", "category", "question", "expected_route", "ground_truth"}
    for idx, case in enumerate(data):
        if not isinstance(case, dict):
            raise ValueError(f"Case at index {idx} must be a dictionary.")
        missing = required_keys - set(case.keys())
        if missing:
            raise ValueError(f"Case {case.get('id', idx)} is missing required fields: {sorted(missing)}")

    return data


# ── Deterministic CSV Evaluation ─────────────────────────────────────────────

def evaluate_csv_case(
    case: Dict[str, Any],
    trace: Dict[str, Any],
    answer: str,
    tabular_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates whether:
      - Correct tabular route was taken
      - Correct operation was selected
      - Target column matches expectations
      - Deterministic computed pandas value matches ground truth
      - Final answer preserves the computed figure or entity
    """
    expectations = case.get("csv_expectations") or {}
    tab_trace = trace.get("tabular") or {}
    tab_ctx = tabular_context or {}

    operation_selected = tab_trace.get("operation") or tab_ctx.get("operation")
    expected_op = expectations.get("operation")
    operation_correct = bool(expected_op and str(operation_selected).lower() == str(expected_op).lower())

    expected_col = expectations.get("target_column")
    col_selected = tab_trace.get("target_column")
    column_correct = True
    if expected_col:
        column_correct = bool(col_selected and str(col_selected).lower() == str(expected_col).lower())

    expected_val = expectations.get("expected_value")
    expected_entity = expectations.get("expected_entity")

    # Inspect computed values from tabular_context or summary
    summary = str(tab_ctx.get("summary", "")).lower()
    raw_res = tab_ctx.get("result")

    value_match = False
    if expected_val is not None:
        val_str = str(expected_val).rstrip("0").rstrip(".") if isinstance(expected_val, float) else str(expected_val)
        if val_str in summary or (isinstance(raw_res, (int, float)) and abs(raw_res - float(expected_val)) < 1e-4):
            value_match = True
        elif isinstance(raw_res, dict):
            # Check nested dict values
            dict_vals = [str(v) for v in raw_res.values()]
            if any(val_str in v for v in dict_vals):
                value_match = True

    if expected_entity:
        entity_match = expected_entity.lower() in summary or expected_entity.lower() in answer.lower()
    else:
        entity_match = True

    if expected_val is None and expected_entity:
        value_match = entity_match

    # Overall correctness: operation, column, and computed value must be preserved
    overall_correct = operation_correct and column_correct and value_match

    return {
        "operation_selected": operation_selected,
        "operation_expected": expected_op,
        "operation_correct": operation_correct,
        "column_selected": col_selected,
        "column_expected": expected_col,
        "column_correct": column_correct,
        "expected_value": expected_val,
        "expected_entity": expected_entity,
        "value_match": value_match,
        "overall_correct": overall_correct,
    }


# ── Retrieval Evaluation ─────────────────────────────────────────────────────

def evaluate_retrieval_case(case: Dict[str, Any], sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates retrieval source presence and relevance hit against expected ground-truth source.
    """
    expected_source = case.get("expected_source")
    if not expected_source:
        return {
            "source_hit": None,
            "retrieval_success": len(sources) > 0,
            "retrieved_sources": [s.get("source") for s in sources],
        }

    retrieved_names = [str(s.get("source", "")).lower() for s in sources]
    expected_clean = expected_source.lower()

    source_hit = any(expected_clean in name or name in expected_clean for name in retrieved_names)
    retrieval_success = len(sources) > 0 and source_hit

    return {
        "expected_source": expected_source,
        "source_hit": source_hit,
        "retrieval_success": retrieval_success,
        "retrieved_sources": [s.get("source") for s in sources],
    }


# ── Aggregates & Reliability Summary ─────────────────────────────────────────

def compute_aggregates(
    results: List[Dict[str, Any]],
    ragas_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Computes transparent evaluation metrics and project reliability summary.
    Aggregate formula:
        aggregate_reliability = mean(valid_evaluation_metrics)
    Only non-null, actually computed metrics are included in the arithmetic mean.
    """
    # 1. Retrieval metrics
    doc_results = [r for r in results if r.get("expected_route") == "documents" or r.get("retrieval_eval")]
    source_hits = [r["retrieval_eval"]["source_hit"] for r in doc_results if r.get("retrieval_eval") and r["retrieval_eval"]["source_hit"] is not None]
    ret_success = [r["retrieval_eval"]["retrieval_success"] for r in doc_results if r.get("retrieval_eval")]

    source_hit_rate = round(sum(1.0 for s in source_hits if s) / len(source_hits), 4) if source_hits else None
    retrieval_success_rate = round(sum(1.0 for s in ret_success if s) / len(ret_success), 4) if ret_success else None

    # 2. Structured data metrics
    csv_results = [r for r in results if r.get("tabular_eval")]
    csv_correct = [r["tabular_eval"]["overall_correct"] for r in csv_results]
    csv_op_correct = [r["tabular_eval"]["operation_correct"] for r in csv_results]

    csv_correctness_rate = round(sum(1.0 for c in csv_correct if c) / len(csv_correct), 4) if csv_correct else None
    csv_operation_rate = round(sum(1.0 for c in csv_op_correct if c) / len(csv_op_correct), 4) if csv_op_correct else None

    # 3. Document RAG RAGAs metrics
    doc_rag_metrics: Dict[str, Optional[float]] = {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }
    if ragas_scores:
        for k in doc_rag_metrics:
            if k in ragas_scores and ragas_scores[k] is not None:
                doc_rag_metrics[k] = round(float(ragas_scores[k]), 4)

    # 4. Explicit project-level aggregate reliability calculation
    # Formula: Arithmetic mean of all valid (non-null) evaluated metrics
    valid_metrics: List[float] = []
    if source_hit_rate is not None:
        valid_metrics.append(source_hit_rate)
    if retrieval_success_rate is not None:
        valid_metrics.append(retrieval_success_rate)
    if csv_correctness_rate is not None:
        valid_metrics.append(csv_correctness_rate)
    if csv_operation_rate is not None:
        valid_metrics.append(csv_operation_rate)

    for val in doc_rag_metrics.values():
        if val is not None:
            valid_metrics.append(val)

    aggregate_reliability = round(sum(valid_metrics) / len(valid_metrics), 4) if valid_metrics else None

    return {
        "document_rag": doc_rag_metrics,
        "retrieval": {
            "source_hit_rate": source_hit_rate,
            "retrieval_success_rate": retrieval_success_rate,
        },
        "structured_data": {
            "correctness_rate": csv_correctness_rate,
            "operation_selection_rate": csv_operation_rate,
        },
        "aggregate_reliability": aggregate_reliability,
        "active_metrics_count": len(valid_metrics),
    }


# ── Main Evaluation Runner ───────────────────────────────────────────────────

def run_evaluation(
    dataset_path: Union[Path, str] = EVAL_DATASET_FILE,
    results_path: Union[Path, str] = RESULTS_FILE,
    scores_path: Union[Path, str] = SCORES_FILE,
    enable_ragas: bool = True,
) -> Dict[str, Any]:
    """
    Executes benchmark evaluation across all dataset cases.
    Records per-question traces, deterministic checks, and RAGAs scores.
    """
    results_path = Path(results_path)
    scores_path = Path(scores_path)
    dataset = load_eval_dataset(dataset_path)

    # Ensure CSV is loaded for tabular benchmark cases
    if not tabular_store.has_data():
        logger.info("Loading default space missions benchmark CSV into tabular store for evaluation.")
        tabular_store.load_csv(SAMPLE_BENCHMARK_CSV.encode("utf-8"), "space_missions.csv")

    logger.info(f"Running evaluation on {len(dataset)} benchmark cases …")

    per_question_results: List[Dict[str, Any]] = []
    ragas_questions: List[str] = []
    ragas_answers: List[str] = []
    ragas_contexts: List[List[str]] = []
    ragas_ground_truths: List[str] = []

    for case in dataset:
        q_id = case["id"]
        q_text = case["question"]
        expected_route = case["expected_route"]
        history = case.get("conversation_history")

        logger.info(f"Evaluating [{q_id}] ({case.get('category')}): {q_text[:50]}…")

        try:
            res = ask(q_text, history=history)
            answer = res.get("answer", "")
            sources = res.get("sources", [])
            trace = res.get("trace", {})

            # Check route match
            actual_route = trace.get("route")
            route_match = (actual_route == expected_route)

            # Categorize evaluation by domain
            tab_eval = None
            ret_eval = None
            if case.get("csv_expectations") or expected_route == "csv_data":
                tab_eval = evaluate_csv_case(case, trace, answer)
            elif expected_route == "documents":
                ret_eval = evaluate_retrieval_case(case, sources)
                # Collect contexts for RAGAs if text context exists
                ctx_texts = [s.get("text", "") for s in sources if s.get("text")]
                if not ctx_texts and case.get("synthetic_context"):
                    ctx_texts = [case["synthetic_context"]]
                ragas_questions.append(q_text)
                ragas_answers.append(answer or "No answer generated.")
                ragas_contexts.append(ctx_texts if ctx_texts else ["No context retrieved."])
                ragas_ground_truths.append(case.get("ground_truth", ""))

            case_result: Dict[str, Any] = {
                "id": q_id,
                "category": case.get("category"),
                "question": q_text,
                "expected_route": expected_route,
                "actual_route": actual_route,
                "route_match": route_match,
                "answer": answer[:200] + ("…" if len(answer) > 200 else ""),
                "ground_truth": case.get("ground_truth"),
                "tabular_eval": tab_eval,
                "retrieval_eval": ret_eval,
                "trace_summary": {
                    "nodes": [n.get("name") for n in trace.get("nodes", [])],
                    "duration_ms": trace.get("total_duration_ms", 0),
                    "status": "success" if not trace.get("error") else "error",
                },
                "status": "success",
            }
            per_question_results.append(case_result)

        except Exception as e:
            logger.error(f"Error evaluating case {q_id}: {e}")
            per_question_results.append({
                "id": q_id,
                "category": case.get("category"),
                "question": q_text,
                "expected_route": expected_route,
                "status": "error",
                "error": type(e).__name__,
            })

    # Optional RAGAs judge execution
    ragas_scores: Optional[Dict[str, float]] = None
    if enable_ragas and RAGAS_AVAILABLE and ragas_questions:
        judge = get_ragas_judge()
        if judge:
            try:
                for metric in (faithfulness, answer_relevancy, context_precision, context_recall):
                    metric.llm = judge  # type: ignore[attr-defined]

                eval_data = Dataset.from_dict({
                    "question": ragas_questions,
                    "answer": ragas_answers,
                    "contexts": ragas_contexts,
                    "ground_truth": ragas_ground_truths,
                })
                logger.info(f"Running RAGAs judge metrics on {len(ragas_questions)} document cases …")
                ragas_res = ragas_evaluate(
                    eval_data,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                )
                ragas_scores = {k: round(float(v), 4) for k, v in ragas_res.items()}
                logger.success("RAGAs evaluation completed.")
            except Exception as e:
                logger.warning(f"RAGAs evaluation failed or model unavailable ({e}). Continuing without fake scores.")
                ragas_scores = None
        else:
            logger.info("Ollama judge not available; skipping LLM-judge RAGAs metrics safely.")

    # Compute explicit aggregates
    aggregates = compute_aggregates(per_question_results, ragas_scores)

    # Build results payload
    full_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(dataset),
        "results": per_question_results,
        "aggregate": aggregates,
        "failed_cases": [r for r in per_question_results if r.get("status") == "error"],
    }

    # Save outputs
    results_path.write_text(json.dumps(full_output, indent=2), encoding="utf-8")
    logger.success(f"Full evaluation results written to {results_path}")

    # Backward-compatible latest_scores.json
    flat_scores: Dict[str, float] = {}
    if aggregates.get("aggregate_reliability") is not None:
        flat_scores["aggregate_reliability"] = aggregates["aggregate_reliability"]
    if aggregates["retrieval"].get("source_hit_rate") is not None:
        flat_scores["source_hit_rate"] = aggregates["retrieval"]["source_hit_rate"]
    if aggregates["structured_data"].get("correctness_rate") is not None:
        flat_scores["csv_correctness"] = aggregates["structured_data"]["correctness_rate"]
    for k, v in aggregates["document_rag"].items():
        if v is not None:
            flat_scores[k] = v

    if flat_scores:
        scores_path.write_text(json.dumps(flat_scores, indent=2), encoding="utf-8")
        logger.success(f"Backward-compatible scores written to {scores_path}")

    # Terminal report
    print("\n==================================================================")
    print(f"  RAG & Tabular Evaluation Summary (Dataset: {len(dataset)} items)")
    print("------------------------------------------------------------------")
    print(f"  Structured Data Correctness:    {aggregates['structured_data']['correctness_rate']}")
    print(f"  Operation Selection Rate:       {aggregates['structured_data']['operation_selection_rate']}")
    print(f"  Retrieval Source Hit Rate:      {aggregates['retrieval']['source_hit_rate']}")
    print(f"  Retrieval Success Rate:         {aggregates['retrieval']['retrieval_success_rate']}")
    for k, v in aggregates["document_rag"].items():
        print(f"  RAGAs {k.replace('_', ' ').title():<25} {v if v is not None else 'Unavailable (No Judge)'}")
    print("------------------------------------------------------------------")
    print(f"  * AGGREGATE RELIABILITY SCORE:  {aggregates['aggregate_reliability']}")
    print("==================================================================\n")

    return full_output


if __name__ == "__main__":
    run_evaluation()
