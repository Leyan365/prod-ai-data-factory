"""Tests for the FastAPI web interface and REST API."""

import asyncio
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from training_data_bot.web.app import create_app


@pytest.fixture
def client(tmp_path: Path):
    storage_dir = tmp_path / "storage"
    app = create_app(storage_dir=str(storage_dir))
    with TestClient(app) as test_client:
        yield test_client


def test_serve_index(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "Training Data Factory" in response.text
    assert "/static/app.js" in response.text


def test_static_assets(client: TestClient):
    css_res = client.get("/static/style.css")
    assert css_res.status_code == 200
    assert "glass-panel" in css_res.text

    js_res = client.get("/static/app.js")
    assert js_res.status_code == 200
    assert "addEventListener" in js_res.text


def test_api_status(client: TestClient):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["version"] == "0.2.0"
    assert data["default_provider"] == "mock"
    assert "jsonl" in data["supported_exports"]
    assert "csv" in data["supported_exports"]


def test_api_jobs_and_datasets_empty(client: TestClient):
    res_jobs = client.get("/api/jobs")
    assert res_jobs.status_code == 200
    assert isinstance(res_jobs.json(), list)

    res_ds = client.get("/api/datasets")
    assert res_ds.status_code == 200
    assert isinstance(res_ds.json(), list)


def test_process_invalid_file_extension(client: TestClient):
    files = [("files", ("bad_file.exe", b"binarycontent", "application/octet-stream"))]
    res = client.post("/api/process", files=files)
    assert res.status_code == 400
    assert "not supported" in res.json()["detail"]


def test_process_empty_files(client: TestClient):
    files = [("files", ("empty.txt", b"", "text/plain"))]
    res = client.post("/api/process", files=files)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"]


def test_process_and_export_lifecycle(client: TestClient):
    sample_text = (
        "Artificial Intelligence in Healthcare:\n"
        "Machine learning models are increasingly used for diagnostic assistance, "
        "medical imaging analysis, and personalized treatment planning. "
        "Regulatory adherence and ethical data curation are critical components."
    )
    files = [("files", ("sample.txt", sample_text.encode("utf-8"), "text/plain"))]
    data = {
        "tasks": "qa,summarization",
        "provider": "mock",
        "quality_filter": "true",
        "split": "false",
        "format": "jsonl",
    }

    res = client.post("/api/process", files=files, data=data)
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    assert job_id

    # Poll until job completes
    import time
    completed = False
    dataset_id = None
    for _ in range(30):
        time.sleep(0.3)
        job_res = client.get(f"/api/jobs/{job_id}")
        assert job_res.status_code == 200
        job_data = job_res.json()
        if job_data["status"] in ("completed", "partial"):
            completed = True
            dataset_id = job_data["dataset_id"]
            break
        elif job_data["status"] == "failed":
            pytest.fail(f"Job failed: {job_data.get('error_message')}")

    assert completed
    assert dataset_id

    # Inspect dataset details
    ds_res = client.get(f"/api/datasets/{dataset_id}")
    assert ds_res.status_code == 200
    ds_data = ds_res.json()
    assert ds_data["example_count"] > 0
    assert len(ds_data["examples"]) > 0

    # Test export as JSONL
    export_jsonl = client.get(f"/api/datasets/{dataset_id}/export?format=jsonl")
    assert export_jsonl.status_code == 200
    assert len(export_jsonl.content) > 0

    # Test export as CSV
    export_csv = client.get(f"/api/datasets/{dataset_id}/export?format=csv")
    assert export_csv.status_code == 200
    assert len(export_csv.content) > 0

    # Test export as split zip
    export_split = client.get(f"/api/datasets/{dataset_id}/export?format=jsonl&split=true")
    assert export_split.status_code == 200
    assert export_split.headers["content-type"] == "application/zip"
    assert len(export_split.content) > 0


def test_get_nonexistent_dataset_and_job(client: TestClient):
    res_ds = client.get("/api/datasets/00000000-0000-0000-0000-000000000000")
    assert res_ds.status_code == 404

    res_job = client.get("/api/jobs/00000000-0000-0000-0000-000000000000")
    assert res_job.status_code == 404
