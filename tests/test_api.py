"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["corpus_size"] >= 30

def test_corpus_endpoint():
    response = client.get("/api/corpus")
    assert response.status_code == 200
    papers = response.json()
    assert len(papers) >= 30
    assert any(p["paper_id"] == "flashattention_v1" for p in papers)

def test_query_endpoint():
    payload = {
        "query": "How does FlashAttention avoid memory bottlenecks?",
        "retrieval_mode": "hybrid",
        "top_k": 3,
        "audit_faithfulness": True
    }
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["retrieved_chunks"]) >= 1
    assert data["faithfulness"] is not None

def test_compare_endpoint():
    payload = {
        "paper_ids": ["flashattention_v1", "flashattention_v2"]
    }
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary_matrix" in data
    assert len(data["summary_matrix"]) >= 3
