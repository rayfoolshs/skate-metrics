from __future__ import annotations

import time

from fastapi.testclient import TestClient

from skatemetrics import server


def test_index_has_full_interface():
    client = TestClient(server.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Skate Metrics" in response.text
    assert "Sessions" in response.text
    assert "Record on this device" in response.text


def test_demo_job_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "BASE_OUT", tmp_path)
    server._load_jobs()
    client = TestClient(server.app)

    response = client.post("/api/demo")
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    job = {}
    for _ in range(120):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "error"}:
            break
        time.sleep(0.1)

    assert job["status"] == "done", job.get("error")
    assert job["jump_count"] == 1
    assert job["jumps"][0]["half_turns"] == 1.5
    assert job["report_url"].endswith("/report")
    assert job["skeleton_url"].endswith("/skeleton.mp4")

    assert client.get(f"/jobs/{job_id}/report").status_code == 200
    assert client.get(f"/jobs/{job_id}/overlay.mp4").status_code == 200
    assert client.get(f"/jobs/{job_id}/skeleton.mp4").status_code == 200
    assert client.get(f"/jobs/{job_id}/metrics.json").status_code == 200

    listed = client.get("/api/jobs").json()
    assert any(item["id"] == job_id for item in listed)

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
