from __future__ import annotations

from fastapi.testclient import TestClient

from skatemetrics import server
from skatemetrics.youtube import is_youtube_url, normalize_url


def test_youtube_url_detection():
    assert is_youtube_url("https://www.youtube.com/watch?v=abc")
    assert is_youtube_url("https://youtu.be/abc")
    assert is_youtube_url("youtube.com/shorts/abc")
    assert is_youtube_url("https://m.youtube.com/watch?v=abc")
    assert not is_youtube_url("https://example.com/watch?v=abc")
    assert not is_youtube_url("not a url")


def test_normalize_url_adds_scheme():
    assert normalize_url("youtu.be/abc") == "https://youtu.be/abc"


def test_youtube_endpoint_validates_input():
    client = TestClient(server.app)
    response = client.post("/api/youtube", json={"url": "https://example.com/abc"})
    assert response.status_code == 400

    response = client.post(
        "/api/youtube",
        json={"url": "https://youtu.be/abc", "pose": "nope"},
    )
    assert response.status_code == 400

    response = client.post(
        "/api/youtube",
        json={"url": "https://youtu.be/abc", "cookies": "netscape"},
    )
    assert response.status_code == 400
