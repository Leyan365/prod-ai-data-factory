"""Service layer coordinating TrainingDataBot with the web interface."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4
import zipfile

from .. import __version__
from ..ai import AIClient
from ..bot import TrainingDataBot
from ..core.config import settings
from ..core.exceptions import TrainingDataBotError
from ..core.models import ExportFormat, ProcessingJob, ProcessingStatus, TaskType
from ..storage import DatasetExporter


ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm", ".json", ".csv"
}
SUPPORTED_FORMATS = tuple(sorted(item.value for item in DatasetExporter.SUPPORTED_FORMATS))

TASK_TYPE_MAP: Dict[str, TaskType] = {
    "qa": TaskType.QA_GENERATION,
    "qa_generation": TaskType.QA_GENERATION,
    "classification": TaskType.CLASSIFICATION,
    "summarization": TaskType.SUMMARIZATION,
    "named_entity_recognition": TaskType.NER,
    "ner": TaskType.NER,
    "red_teaming": TaskType.RED_TEAMING,
    "instruction_response": TaskType.INSTRUCTION_RESPONSE,
}


def _load_env_gemini_key() -> Optional[str]:
    """Check environment or local .env file for GEMINI_API_KEY."""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    for candidate in [Path(".env"), Path(__file__).resolve().parents[3] / ".env"]:
        if candidate.exists():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "GEMINI_API_KEY":
                            val = v.strip().strip('"').strip("'")
                            if val:
                                os.environ["GEMINI_API_KEY"] = val
                                return val
            except Exception:
                pass
    return None


class WebService:
    """Manages bot instance, background jobs, and web actions."""

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        config: Dict[str, Any] = {}
        if self.storage_dir:
            config["storage_dir"] = self.storage_dir
        self.bot = TrainingDataBot(config=config)
        self.active_jobs: Dict[str, Dict[str, Any]] = {}

    async def get_status(self) -> Dict[str, Any]:
        """Return system status and storage metadata."""
        datasets = await self.bot.list_persisted_datasets()
        jobs = await self.bot.list_persisted_jobs()
        return {
            "version": __version__,
            "default_provider": settings.default_ai_provider,
            "supported_exports": list(SUPPORTED_FORMATS),
            "storage_dir": str(self.bot.db_manager.storage_dir),
            "persisted_datasets": len(datasets),
            "persisted_jobs": len(jobs),
        }

    async def start_processing_job(
        self,
        files: List[Tuple[str, bytes]],
        task_types: List[str],
        provider: str = "mock",
        model: str = "gemini-1.5-flash",
        quality_filter: bool = True,
        split: bool = False,
        export_format: str = "jsonl",
    ) -> str:
        """Validate uploaded files and spawn an async processing job."""
        if not files:
            raise ValueError("No files provided for processing")

        # Validate file extensions
        for filename, _ in files:
            ext = Path(filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise ValueError(
                    f"File type '{ext}' is not supported. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                )

        if export_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{export_format}'. Supported: {', '.join(SUPPORTED_FORMATS)}")

        if provider not in ("mock", "gemini"):
            raise ValueError(f"Unsupported provider: {provider}")

        if provider == "gemini" and not _load_env_gemini_key():
            raise ValueError("GEMINI_API_KEY is required in .env or environment for Gemini provider")

        job_id = str(uuid4())
        job_info: Dict[str, Any] = {
            "id": job_id,
            "name": f"Web Process ({len(files)} files)",
            "status": "queued",
            "stage": "queued",
            "total_items": 0,
            "processed_items": 0,
            "failed_items": 0,
            "error_message": None,
            "dataset_id": None,
            "export_path": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_data": {},
        }
        self.active_jobs[job_id] = job_info

        # Run the job asynchronously in background
        asyncio.create_task(
            self._execute_job(
                job_id=job_id,
                files=files,
                task_types=task_types,
                provider=provider,
                model=model,
                quality_filter=quality_filter,
                split=split,
                export_format=export_format,
            )
        )

        return job_id

    async def _execute_job(
        self,
        job_id: str,
        files: List[Tuple[str, bytes]],
        task_types: List[str],
        provider: str,
        model: str,
        quality_filter: bool,
        split: bool,
        export_format: str,
    ) -> None:
        """Worker task processing files and tracking progress."""
        upload_dir = Path(".tmp") / "web_uploads" / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        job_info = self.active_jobs[job_id]

        try:
            # 1. Save files to temp upload directory
            job_info["status"] = "processing"
            job_info["stage"] = "saving_files"
            for filename, data in files:
                safe_name = Path(filename).name
                file_path = upload_dir / safe_name
                file_path.write_bytes(data)

            # 2. Configure provider
            if provider == "mock":
                self.bot.ai_client = AIClient.from_env("mock")
            elif provider == "gemini":
                self.bot.ai_client = AIClient.from_env("gemini", model=model)

            # 3. Parse tasks
            parsed_tasks = []
            if task_types:
                for t in task_types:
                    key = t.lower().strip()
                    if key in TASK_TYPE_MAP:
                        parsed_tasks.append(TASK_TYPE_MAP[key])
                    else:
                        parsed_tasks.append(TaskType(t))
            else:
                parsed_tasks = None

            # 4. Load documents
            job_info["stage"] = "loading_documents"
            documents = await self.bot.load_documents(upload_dir)
            if not documents:
                raise TrainingDataBotError("No valid documents could be loaded from uploads")

            # 5. Process documents
            job_info["stage"] = "generating_examples"
            dataset = await self.bot.process_documents(
                documents=documents,
                task_types=parsed_tasks,
                quality_filter=quality_filter,
            )

            # 6. Export dataset
            job_info["stage"] = "exporting_dataset"
            export_dir = Path("output") / "web_exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            target_export = export_dir / f"{dataset.id}.{export_format}"

            export_path = await self.bot.export_dataset(
                dataset=dataset,
                output_path=target_export,
                format=ExportFormat(export_format),
                split_data=split,
            )

            # Determine final status
            processing_status = dataset.metadata.get("processing_status", "completed")
            job_info["status"] = processing_status
            job_info["stage"] = "finished"
            job_info["dataset_id"] = str(dataset.id)
            job_info["export_path"] = str(export_path)
            job_info["output_data"] = {
                "dataset_id": str(dataset.id),
                "examples_generated": len(dataset.examples),
                "quality_filtered": quality_filter,
                "split": split,
                "format": export_format,
            }

        except Exception as exc:
            job_info["status"] = "failed"
            job_info["stage"] = "failed"
            job_info["error_message"] = str(exc)
        finally:
            # Clean up temp files
            if upload_dir.exists():
                shutil.rmtree(upload_dir, ignore_errors=True)

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job status from memory or persistent storage."""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]

        try:
            persisted = await self.bot.load_job(job_id)
            return {
                "id": str(persisted.id),
                "name": persisted.name,
                "status": persisted.status.value if hasattr(persisted.status, "value") else str(persisted.status),
                "stage": "finished",
                "total_items": persisted.total_items,
                "processed_items": persisted.processed_items,
                "failed_items": persisted.failed_items,
                "error_message": persisted.error_message,
                "dataset_id": persisted.output_data.get("dataset_id") if persisted.output_data else None,
                "export_path": None,
                "created_at": persisted.created_at.isoformat() if hasattr(persisted.created_at, "isoformat") else str(persisted.created_at),
                "output_data": persisted.output_data or {},
            }
        except Exception:
            return None

    async def list_jobs(self) -> List[Dict[str, Any]]:
        """List active and persisted jobs."""
        seen_ids = set()
        results = []

        # Active / recently cached jobs first
        for job_id, info in self.active_jobs.items():
            results.append(info)
            seen_ids.add(job_id)

        # Persisted jobs
        persisted = await self.bot.list_persisted_jobs()
        for p in persisted:
            pid = str(p.id)
            if pid not in seen_ids:
                results.append({
                    "id": pid,
                    "name": p.name,
                    "status": p.status.value if hasattr(p.status, "value") else str(p.status),
                    "stage": "finished",
                    "total_items": p.total_items,
                    "processed_items": p.processed_items,
                    "failed_items": p.failed_items,
                    "error_message": p.error_message,
                    "dataset_id": p.output_data.get("dataset_id") if p.output_data else None,
                    "export_path": None,
                    "created_at": p.created_at.isoformat() if hasattr(p.created_at, "isoformat") else str(p.created_at),
                    "output_data": p.output_data or {},
                })
                seen_ids.add(pid)

        return results

    async def list_datasets(self) -> List[Dict[str, Any]]:
        """List persisted datasets."""
        datasets = await self.bot.list_persisted_datasets()
        results = []
        for ds in datasets:
            results.append({
                "id": str(ds.id),
                "name": ds.name,
                "description": ds.description,
                "example_count": len(ds.examples),
                "created_at": ds.created_at.isoformat() if hasattr(ds.created_at, "isoformat") else str(ds.created_at),
                "export_format": ds.export_format.value if getattr(ds, "export_format", None) else None,
                "export_path": str(ds.export_path) if getattr(ds, "export_path", None) else None,
            })
        return results

    async def get_dataset(self, dataset_id: str, limit: int = 50, offset: int = 0) -> Optional[Dict[str, Any]]:
        """Load dataset with paginated examples."""
        try:
            ds = await self.bot.load_dataset(dataset_id)
        except Exception:
            return None

        examples = []
        sliced = ds.examples[offset : offset + limit]
        for ex in sliced:
            examples.append({
                "input_text": ex.input_text,
                "output_text": ex.output_text,
                "task_type": ex.task_type.value if hasattr(ex.task_type, "value") else str(ex.task_type),
                "quality_scores": ex.quality_scores or {},
                "quality_approved": ex.quality_approved,
            })

        return {
            "id": str(ds.id),
            "name": ds.name,
            "description": ds.description,
            "example_count": len(ds.examples),
            "created_at": ds.created_at.isoformat() if hasattr(ds.created_at, "isoformat") else str(ds.created_at),
            "metadata": ds.metadata or {},
            "examples": examples,
        }

    async def export_persisted_dataset(
        self,
        dataset_id: str,
        export_format: str = "jsonl",
        split: bool = False,
    ) -> Path:
        """Export dataset to disk and package into zip if split."""
        if export_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {export_format}")

        ds = await self.bot.load_dataset(dataset_id)
        export_root = Path("output") / "web_exports"
        export_root.mkdir(parents=True, exist_ok=True)

        if split:
            split_dir = export_root / f"{ds.id}_split_{export_format}"
            await self.bot.export_dataset(
                dataset=ds,
                output_path=split_dir,
                format=ExportFormat(export_format),
                split_data=True,
            )
            # Create zip archive of the split folder
            zip_target = export_root / f"dataset_{ds.id}_split_{export_format}.zip"
            with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(split_dir):
                    for file in files:
                        full_path = Path(root) / file
                        rel_path = full_path.relative_to(split_dir)
                        zf.write(full_path, arcname=str(rel_path))
            return zip_target
        else:
            target_path = export_root / f"dataset_{ds.id}.{export_format}"
            exported_path = await self.bot.export_dataset(
                dataset=ds,
                output_path=target_path,
                format=ExportFormat(export_format),
                split_data=False,
            )
            return exported_path

    async def cleanup(self) -> None:
        """Clean up bot resources."""
        await self.bot.cleanup()
