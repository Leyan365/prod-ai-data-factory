"""FastAPI application for Training Data Bot web interface."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import DatasetDetail, DatasetSummary, JobSummary, ProcessResponse, StatusResponse
from .service import WebService


STATIC_DIR = Path(__file__).parent / "static"


def create_app(storage_dir: Optional[str] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    service = WebService(storage_dir=storage_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await service.cleanup()

    app = FastAPI(
        title="Training Data Factory",
        description="Modern web interface for LLM training data curation",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Attach service to app state for easy test inspection
    app.state.service = service

    # Mount static assets
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = STATIC_DIR / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Frontend assets not found")
        return FileResponse(str(index_file))

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status():
        return await service.get_status()

    @app.post("/api/process", response_model=ProcessResponse)
    async def process_files(
        files: List[UploadFile] = File(...),
        tasks: str = Form("qa,summarization,classification"),
        provider: str = Form("mock"),
        model: str = Form("gemini-1.5-flash"),
        quality_filter: bool = Form(True),
        split: bool = Form(False),
        format: str = Form("jsonl"),
    ):
        file_payloads = []
        for upload in files:
            content = await upload.read()
            if content:
                file_payloads.append((upload.filename or "uploaded_file", content))

        if not file_payloads:
            raise HTTPException(status_code=400, detail="No files uploaded or files were empty")

        task_list = [t.strip() for t in tasks.split(",") if t.strip()]

        try:
            job_id = await service.start_processing_job(
                files=file_payloads,
                task_types=task_list,
                provider=provider,
                model=model,
                quality_filter=quality_filter,
                split=split,
                export_format=format,
            )
            return ProcessResponse(
                job_id=job_id,
                status="queued",
                message="Processing job started successfully",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to start job: {exc}")

    @app.get("/api/jobs", response_model=List[JobSummary])
    async def list_jobs():
        jobs = await service.list_jobs()
        return [JobSummary(**j) for j in jobs]

    @app.get("/api/jobs/{job_id}", response_model=JobSummary)
    async def get_job(job_id: str):
        job = await service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return JobSummary(**job)

    @app.get("/api/datasets", response_model=List[DatasetSummary])
    async def list_datasets():
        datasets = await service.list_datasets()
        return [DatasetSummary(**d) for d in datasets]

    @app.get("/api/datasets/{dataset_id}", response_model=DatasetDetail)
    async def get_dataset(
        dataset_id: str,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        dataset = await service.get_dataset(dataset_id, limit=limit, offset=offset)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        return DatasetDetail(**dataset)

    @app.get("/api/datasets/{dataset_id}/export")
    async def export_dataset(
        dataset_id: str,
        format: str = Query("jsonl", pattern="^(jsonl|csv)$"),
        split: bool = Query(False),
    ):
        try:
            export_path = await service.export_persisted_dataset(
                dataset_id=dataset_id,
                export_format=format,
                split=split,
            )
            media_type = "application/zip" if split else (
                "application/x-ndjson" if format == "jsonl" else "text/csv"
            )
            return FileResponse(
                path=str(export_path),
                filename=export_path.name,
                media_type=media_type,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Export failed: {exc}")

    return app


def run_server() -> None:
    """Console script entrypoint for training-data-bot-web."""
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="training-data-bot-web",
        description="Run the Training Data Bot modern web dashboard.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind to. Defaults to 127.0.0.1 (local only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on. Defaults to 8765.",
    )
    parser.add_argument(
        "--storage-dir",
        default=None,
        help="Custom storage directory path.",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" ⚡ Training Data Factory - Modern Web Interface")
    print("=" * 60)
    print(f" URL:         http://{args.host}:{args.port}")
    print(f" Bind:        {args.host}")
    print(f" Port:        {args.port}")
    print(f" Storage:     {args.storage_dir or 'output/storage (default)'}")
    print(" Press Ctrl+C to stop the server.")
    print("=" * 60 + "\n")

    app = create_app(storage_dir=args.storage_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    run_server()
