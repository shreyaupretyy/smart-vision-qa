"""
API route tests.
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_upload_video_no_file():
    """Test video upload without file."""
    response = client.post("/api/videos/upload")
    assert response.status_code == 422

def test_get_nonexistent_video():
    """Test getting non-existent video."""
    response = client.get("/api/videos/nonexistent_id")
    assert response.status_code == 404

def test_query_without_video():
    """Test query without video ID."""
    response = client.post("/api/query", json={
        "query": "test query"
    })
    assert response.status_code == 422

def test_detect_without_video():
    """Test detection without video ID."""
    response = client.post("/api/detect", json={
        "classes": ["person"]
    })
    assert response.status_code == 422

def test_search_invalid_params():
    """Test search with invalid parameters."""
    response = client.post("/api/search", json={
        "video_id": "test",
        "top_k": -1  # Invalid top_k
    })
    assert response.status_code in [400, 422]
