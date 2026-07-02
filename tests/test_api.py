"""Tests for the FastAPI endpoints in llmsearch/api.py.

These tests use FastAPI's dependency-override mechanism together with
unittest.mock so that no real LLM, embedding model, or config file is
needed.  They cover:

  * health-check endpoint
  * query input validation (min/max length → HTTP 422)
  * label validation (unknown label → HTTP 404)
  * optional API key authentication (HTTP 401 when key is wrong)
  * /labels endpoint
  * /rag_chunks endpoint (mocked retrieval)
"""

import os
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Ensure env vars exist before the api module is imported so that the module-
# level code in api.py doesn't blow up (the actual values don't matter because
# we override all dependencies that use them).
# ---------------------------------------------------------------------------
os.environ.setdefault("FASTAPI_RAG_CONFIG", "dummy_rag_config.yaml")
os.environ.setdefault("FASTAPI_LLM_CONFIG", "dummy_llm_config.yaml")
# Make sure auth is OFF by default for most tests
os.environ.pop("LLMSEARCH_API_KEY", None)

from llmsearch.api import api_app, get_config, get_llm_bundle_cached  # noqa: E402
from llmsearch.config import ResponseModel  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_config(labels=None):
    """Return a minimal MagicMock that looks like a Config object."""
    cfg = MagicMock()
    cfg.embeddings.labels = labels if labels is not None else []
    cfg.semantic_search = MagicMock()
    cfg.cache_folder = "/tmp/cache"
    cfg.embeddings.embeddings_path = "/tmp/embeddings"
    return cfg


def _make_mock_bundle():
    """Return a minimal MagicMock that looks like an LLMBundle."""
    return MagicMock()


def _make_mock_response(question="test question"):
    """Return a ResponseModel instance with dummy data."""
    return ResponseModel(
        id=uuid4(),
        question=question,
        response="This is a test answer.",
        average_score=0.85,
        semantic_search=[],
        hyde_response="",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """TestClient with both heavy dependencies stubbed out."""
    mock_cfg = _make_mock_config(labels=["docs", "test_label"])
    mock_bundle = _make_mock_bundle()

    api_app.dependency_overrides[get_config] = lambda: mock_cfg
    api_app.dependency_overrides[get_llm_bundle_cached] = lambda: mock_bundle

    with TestClient(api_app, raise_server_exceptions=False) as c:
        yield c

    api_app.dependency_overrides.clear()


@pytest.fixture()
def client_with_auth(monkeypatch):
    """TestClient with API key authentication enabled (key = 'supersecret')."""
    monkeypatch.setenv("LLMSEARCH_API_KEY", "supersecret")

    mock_cfg = _make_mock_config(labels=["docs"])
    mock_bundle = _make_mock_bundle()

    api_app.dependency_overrides[get_config] = lambda: mock_cfg
    api_app.dependency_overrides[get_llm_bundle_cached] = lambda: mock_bundle

    with TestClient(api_app, raise_server_exceptions=False) as c:
        yield c

    api_app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_welcome_message(self, client):
        response = client.get("/")
        body = response.json()
        assert "message" in body
        assert "LLMSearch" in body["message"]


# ---------------------------------------------------------------------------
# Input validation  (FastAPI enforces Query constraints before the handler
# runs, so no RAG mocking is needed for 422 cases)
# ---------------------------------------------------------------------------

class TestInputValidation:
    @pytest.mark.parametrize("endpoint", ["/llm", "/rag_text_response", "/rag_chunks"])
    def test_missing_question_returns_422(self, client, endpoint):
        """Omitting `question` entirely must return 422 Unprocessable Entity."""
        response = client.get(endpoint)
        assert response.status_code == 422

    @pytest.mark.parametrize("endpoint", ["/llm", "/rag_text_response", "/rag_chunks"])
    def test_empty_question_returns_422(self, client, endpoint):
        """An empty string violates min_length=1 and must return 422."""
        response = client.get(endpoint, params={"question": ""})
        assert response.status_code == 422

    @pytest.mark.parametrize("endpoint", ["/llm", "/rag_text_response", "/rag_chunks"])
    def test_too_long_question_returns_422(self, client, endpoint):
        """A question longer than 2000 chars must return 422."""
        long_question = "x" * 2001
        response = client.get(endpoint, params={"question": long_question})
        assert response.status_code == 422

    @pytest.mark.parametrize("endpoint", ["/llm", "/rag_text_response", "/rag_chunks"])
    def test_max_length_boundary_accepted(self, client, endpoint):
        """Exactly 2000 chars is at the boundary and must NOT return 422."""
        boundary_question = "x" * 2000
        # The RAG pipeline is mocked so we just check it isn't a validation error
        with patch("llmsearch.api.get_and_parse_response", return_value=_make_mock_response()), \
             patch("llmsearch.api.get_config", return_value=_make_mock_config()), \
             patch("llmsearch.api.get_relevant_documents", return_value=([], -1.0)):
            response = client.get(endpoint, params={"question": boundary_question})
        assert response.status_code != 422

    @pytest.mark.parametrize("endpoint", ["/llm", "/rag_text_response", "/rag_chunks"])
    def test_min_length_boundary_accepted(self, client, endpoint):
        """A single character is valid (min_length=1)."""
        with patch("llmsearch.api.get_and_parse_response", return_value=_make_mock_response()), \
             patch("llmsearch.api.get_config", return_value=_make_mock_config()), \
             patch("llmsearch.api.get_relevant_documents", return_value=([], -1.0)):
            response = client.get(endpoint, params={"question": "?"})
        assert response.status_code != 422


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------

class TestLabelValidation:
    def test_unknown_label_returns_404(self, client):
        """Supplying a label that doesn't exist must return 404."""
        mock_cfg = _make_mock_config(labels=["valid_label"])
        with patch("llmsearch.api.get_config", return_value=mock_cfg):
            response = client.get("/llm", params={"question": "hello", "label": "nonexistent"})
        assert response.status_code == 404

    def test_unknown_label_simple_endpoint_returns_404(self, client):
        mock_cfg = _make_mock_config(labels=["valid_label"])
        with patch("llmsearch.api.get_config", return_value=mock_cfg):
            response = client.get("/rag_text_response", params={"question": "hello", "label": "nonexistent"})
        assert response.status_code == 404

    def test_valid_label_does_not_return_404(self, client):
        """A known label must pass label validation (may fail at RAG stage, not 404)."""
        mock_cfg = _make_mock_config(labels=["valid_label"])
        with patch("llmsearch.api.get_config", return_value=mock_cfg), \
             patch("llmsearch.api.get_and_parse_response", return_value=_make_mock_response()):
            response = client.get("/llm", params={"question": "hello", "label": "valid_label"})
        assert response.status_code != 404

    def test_empty_label_skips_label_check(self, client):
        """Omitting label (defaults to '') must not trigger label validation at all."""
        with patch("llmsearch.api.get_config", return_value=_make_mock_config(labels=[])), \
             patch("llmsearch.api.get_and_parse_response", return_value=_make_mock_response()):
            response = client.get("/llm", params={"question": "hello"})
        # An empty label is always allowed — the check is `if label and label not in labels`
        assert response.status_code != 404


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------

class TestApiKeyAuth:
    def test_correct_key_is_accepted(self, client_with_auth):
        """Requests with the correct key must pass auth."""
        response = client_with_auth.get("/", headers={"X-Api-Key": "supersecret"})
        assert response.status_code == 200

    def test_wrong_key_returns_401(self, client_with_auth):
        response = client_with_auth.get("/", headers={"X-Api-Key": "wrongkey"})
        assert response.status_code == 401

    def test_missing_key_returns_401(self, client_with_auth):
        """No header at all must also return 401 when auth is enabled."""
        response = client_with_auth.get("/")
        assert response.status_code == 401

    def test_auth_disabled_when_env_var_not_set(self, client):
        """Without LLMSEARCH_API_KEY set, any request (even with no key) must reach the handler."""
        # `client` fixture already pops LLMSEARCH_API_KEY
        response = client.get("/")
        assert response.status_code == 200

    def test_401_detail_message(self, client_with_auth):
        """The 401 response body should include a helpful message."""
        response = client_with_auth.get("/", headers={"X-Api-Key": "bad"})
        assert response.status_code == 401
        detail = response.json().get("detail", "")
        assert "X-Api-Key" in detail or "API key" in detail.lower()

    def test_auth_applied_to_llm_endpoint(self, client_with_auth):
        response = client_with_auth.get("/llm", params={"question": "hello"}, headers={"X-Api-Key": "wrong"})
        assert response.status_code == 401

    def test_auth_applied_to_rag_chunks_endpoint(self, client_with_auth):
        response = client_with_auth.get("/rag_chunks", params={"question": "hello"}, headers={"X-Api-Key": "wrong"})
        assert response.status_code == 401

    def test_auth_applied_to_labels_endpoint(self, client_with_auth):
        response = client_with_auth.get("/labels", headers={"X-Api-Key": "wrong"})
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# /labels endpoint
# ---------------------------------------------------------------------------

class TestLabelsEndpoint:
    def test_returns_list(self, client):
        with patch("llmsearch.api.get_config", return_value=_make_mock_config(labels=["alpha", "beta"])):
            response = client.get("/labels")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_returns_correct_labels(self, client):
        with patch("llmsearch.api.get_config", return_value=_make_mock_config(labels=["alpha", "beta"])):
            response = client.get("/labels")
        assert set(response.json()) == {"alpha", "beta"}

    def test_empty_labels_returns_empty_list(self, client):
        with patch("llmsearch.api.get_config", return_value=_make_mock_config(labels=[])):
            response = client.get("/labels")
        assert response.status_code == 200
        assert response.json() == []


# ---------------------------------------------------------------------------
# /rag_chunks endpoint (mocked retrieval)
# ---------------------------------------------------------------------------

class TestRagChunksEndpoint:
    def test_returns_sources_key(self, client):
        with patch("llmsearch.api.get_relevant_documents", return_value=([], -1.0)), \
             patch("llmsearch.api.get_config", return_value=_make_mock_config()), \
             patch("llmsearch.api.get_llm_bundle_cached", return_value=_make_mock_bundle()):
            response = client.get("/rag_chunks", params={"question": "what is RAG?"})
        assert response.status_code == 200
        assert "sources" in response.json()
