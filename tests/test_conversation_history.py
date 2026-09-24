"""
tests/test_conversation_history.py
──────────────────────────────────
Unit tests for multi-turn conversation history in FastAPI API and LangGraph agent.
Tests:
1. Message conversion from API payload dictionaries to LangChain BaseMessage objects.
2. Handling of single-turn requests with no history (backward compatibility).
3. Handling of two-turn requests where the second question refers to an entity from the first.
4. Contextualization in route_query and rewrite_query.
5. Verification that history is passed end-to-end to ask().
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide mock for fastapi and pydantic if not installed in current environment
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
sys.modules["ingestion"] = MagicMock()
sys.modules["ingestion.ingest"] = mock_ingest

from app.api import parse_history, QueryRequest
from app.agent import _format_history, _to_messages, route_query, rewrite_query


class TestConversationHistory(unittest.TestCase):

    def test_parse_history_empty(self):
        """Test that None or empty list returns an empty list of messages."""
        self.assertEqual(parse_history(None), [])
        self.assertEqual(parse_history([]), [])

    def test_parse_history_conversion(self):
        """Test converting dictionary history to HumanMessage and AIMessage objects."""
        history = [
            {"role": "user", "content": "Tell me about Project Titan at Apple."},
            {"role": "assistant", "content": "Project Titan was Apple's autonomous electric car project."},
            {"role": "user", "content": "Who was leading it?"},
            {"role": "assistant", "content": "It was led by Doug Field and later Kevin Lynch."}
        ]
        parsed = parse_history(history)
        self.assertEqual(len(parsed), 4)
        self.assertIsInstance(parsed[0], HumanMessage)
        self.assertEqual(parsed[0].content, "Tell me about Project Titan at Apple.")
        self.assertIsInstance(parsed[1], AIMessage)
        self.assertEqual(parsed[1].content, "Project Titan was Apple's autonomous electric car project.")
        self.assertIsInstance(parsed[2], HumanMessage)
        self.assertEqual(parsed[2].content, "Who was leading it?")
        self.assertIsInstance(parsed[3], AIMessage)
        self.assertEqual(parsed[3].content, "It was led by Doug Field and later Kevin Lynch.")

    def test_to_messages_helper(self):
        """Test _to_messages handles both BaseMessage instances and dict objects."""
        mixed = [
            HumanMessage(content="Hello"),
            {"role": "assistant", "content": "Hi there!"}
        ]
        messages = _to_messages(mixed)
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)
        self.assertEqual(messages[1].content, "Hi there!")

    def test_format_history(self):
        """Test that _format_history formats prior turns excluding the latest question."""
        messages = [
            HumanMessage(content="What is Project Titan?"),
            AIMessage(content="Project Titan was Apple's autonomous vehicle division."),
            HumanMessage(content="When was it cancelled?")  # current query
        ]
        formatted = _format_history(messages)
        expected = "User: What is Project Titan?\nAssistant: Project Titan was Apple's autonomous vehicle division."
        self.assertEqual(formatted, expected)

    def test_single_turn_no_history_in_route_query(self):
        """Test that a query with no history preserves existing single-turn route_query behavior."""
        state = {
            "messages": [HumanMessage(content="What are the warranty terms?")],
            "query": "What are the warranty terms?",
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
            result_state = route_query(state)

        # Single call for routing, no rephrase call made
        self.assertEqual(mock_llm.invoke.call_count, 1)
        self.assertEqual(result_state["_route"], "documents")
        self.assertEqual(result_state["query"], "What are the warranty terms?")

    def test_two_turn_followup_contextualization_in_route_query(self):
        """
        Test a two-turn conversation where the second question refers to an entity from the first.
        Turn 1: 'Tell me about Project Titan'
        Turn 2: 'When was it cancelled?'
        route_query should resolve 'it' to 'Project Titan' for vector search.
        """
        history_messages = [
            HumanMessage(content="Tell me about Apple's Project Titan."),
            AIMessage(content="Project Titan was Apple's secret autonomous electric vehicle project launched in 2014."),
            HumanMessage(content="When was it cancelled?")
        ]
        state = {
            "messages": history_messages,
            "query": "When was it cancelled?",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
        }

        mock_llm = MagicMock()
        # First call: routing decision -> 'documents'
        # Second call: rephrase query -> 'When was Apple's Project Titan cancelled?'
        resp_route = MagicMock()
        resp_route.content = "documents"
        resp_rephrase = MagicMock()
        resp_rephrase.content = "When was Apple's Project Titan cancelled?"
        mock_llm.invoke.side_effect = [resp_route, resp_rephrase]

        with patch("app.agent.llm", mock_llm):
            result_state = route_query(state)

        self.assertEqual(mock_llm.invoke.call_count, 2)
        self.assertEqual(result_state["_route"], "documents")
        # Query has been contextualized with the entity from Turn 1
        self.assertEqual(result_state["query"], "When was Apple's Project Titan cancelled?")

    def test_rewrite_query_with_conversation_context(self):
        """Test that rewrite_query incorporates conversation history when rewriting a failed retrieval."""
        history_messages = [
            HumanMessage(content="Tell me about Apple's Project Titan."),
            AIMessage(content="Project Titan was Apple's car project."),
            HumanMessage(content="When was it cancelled?")
        ]
        state = {
            "messages": history_messages,
            "query": "When was it cancelled?",
            "rewrite_count": 0,
            "context": [],
            "web_results": [],
            "answer": "",
        }

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Apple Project Titan cancellation date and announcement"
        mock_llm.invoke.return_value = mock_resp

        with patch("app.agent.llm", mock_llm):
            result_state = rewrite_query(state)

        # Check prompt sent to LLM included conversation history
        call_prompt = mock_llm.invoke.call_args[0][0][0].content
        self.assertIn("Conversation History:", call_prompt)
        self.assertIn("Apple's Project Titan", call_prompt)
        self.assertEqual(result_state["query"], "Apple Project Titan cancellation date and announcement")
        self.assertEqual(result_state["rewrite_count"], 1)

    def test_api_query_endpoint_passes_history(self):
        """Test that FastAPI /query endpoint passes parsed history to ask()."""
        from app.api import query

        req_payload = QueryRequest(
            question="When was it cancelled?",
            history=[
                {"role": "user", "content": "Tell me about Project Titan."},
                {"role": "assistant", "content": "Project Titan was an Apple car initiative."}
            ]
        )

        mock_ask = MagicMock()
        mock_ask.return_value = {
            "answer": "Project Titan was cancelled in February 2024. [Source: tech_news.pdf, Page 2]",
            "sources": [{"text": "Cancelled in 2024", "source": "tech_news.pdf", "page": 2}],
            "rewrite_count": 0,
            "used_web": False,
        }

        with patch("app.api.ask", mock_ask):
            response = query(req_payload)

        # Verify ask was called with both question and parsed history list
        self.assertEqual(mock_ask.call_count, 1)
        args, kwargs = mock_ask.call_args
        self.assertEqual(args[0], "When was it cancelled?")
        passed_history = kwargs.get("history")
        self.assertEqual(len(passed_history), 2)
        self.assertIsInstance(passed_history[0], HumanMessage)
        self.assertEqual(passed_history[0].content, "Tell me about Project Titan.")
        self.assertIsInstance(passed_history[1], AIMessage)
        self.assertEqual(passed_history[1].content, "Project Titan was an Apple car initiative.")

        self.assertEqual(response.answer, "Project Titan was cancelled in February 2024. [Source: tech_news.pdf, Page 2]")
        self.assertEqual(len(response.sources), 1)

    def test_api_query_endpoint_single_turn_no_history(self):
        """Test that FastAPI /query endpoint works with history=None (single turn)."""
        from app.api import query

        req_payload = QueryRequest(
            question="What is the refund policy?",
            history=None
        )

        mock_ask = MagicMock()
        mock_ask.return_value = {
            "answer": "Refunds are processed within 14 days. [Source: policy.pdf, Page 1]",
            "sources": [{"text": "14 days refund", "source": "policy.pdf", "page": 1}],
            "rewrite_count": 0,
            "used_web": False,
        }

        with patch("app.api.ask", mock_ask):
            response = query(req_payload)

        self.assertEqual(mock_ask.call_count, 1)
        args, kwargs = mock_ask.call_args
        self.assertEqual(args[0], "What is the refund policy?")
        self.assertEqual(kwargs.get("history"), [])
        self.assertEqual(response.answer, "Refunds are processed within 14 days. [Source: policy.pdf, Page 1]")


if __name__ == "__main__":
    unittest.main()
