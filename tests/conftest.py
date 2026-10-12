import os
import tempfile

import pytest

# Must run before app.config/app.main are imported.
# Keeps test runs from creating a real reviews.db in the project root.
os.environ.setdefault("DATABASE_PATH", os.path.join(tempfile.mkdtemp(prefix="scra-tests-"), "test_reviews.db"))
# Deterministic webhook signing secret for tests; real deployments use the
# production value set in the environment or .env (never committed).
os.environ.setdefault("GITHUB_WEBHOOK_SECRET", "test-webhook-secret")
# Tests must never reach a real LLM, even when the developer's .env selects
# a provider with a valid key: force the default provider and leave its key
# unset so /review takes the documented "not configured" path. Individual
# tests override settings.LLM_PROVIDER / GROQ_API_KEY with mocked HTTP.
os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("GEMINI_API_KEY", "")


@pytest.fixture(scope="session", autouse=True)
def _report_test_db_path():
    # Short-lived diagnostics; visible with pytest -s
    from app.config import settings

    print(f"\n[test db] {settings.DATABASE_PATH}")
    yield
    print(f"\n[test db cleanup] test artifacts left at: {settings.DATABASE_PATH}")
