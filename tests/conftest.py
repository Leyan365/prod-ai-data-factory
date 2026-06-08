"""Shared pytest configuration."""

import sys
from pathlib import Path
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def workspace_tmp(request):
    """Workspace-local temp directory that avoids restricted system temp paths."""

    safe_name = "".join(ch if ch.isalnum() else "_" for ch in request.node.name)
    path = ROOT / ".tmp" / "pytest-fixtures" / f"{safe_name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
