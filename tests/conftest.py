from __future__ import annotations

import sqlite3

import pytest

from app.core.config import PROJECT_ROOT, Settings
from app.database.seed import seed_database
from app.llm.mock import ScriptedProvider
from scripts.generate_documents import generate
from tests.helpers import HashingEmbedder, make_handler


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("acme")
    docs = root / "documents"
    generate(docs)
    db = root / "acme.db"
    seed_database(db)
    settings = Settings(
        documents_dir=docs, index_dir=root / "index", sqlite_path=db, rag_min_similarity=0.25,
        llm_api_key="test", _env_file=None,
    )
    return settings


@pytest.fixture(scope="session")
def embedder():
    return HashingEmbedder()


@pytest.fixture
def ro_truth(env):
    """Independent connection used to compute reference answers."""
    conn = sqlite3.connect(env.sqlite_path)
    yield conn
    conn.close()


@pytest.fixture
def llm_factory():
    return lambda **kw: ScriptedProvider(handler=make_handler(**kw))
