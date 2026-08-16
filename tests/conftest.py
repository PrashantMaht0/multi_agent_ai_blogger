"""
tests/conftest.py
Keeps the suite hermetic: no .env, no credentials, no live LLM or API calls.
Runs before any src import so module-level load_dotenv() calls are already neutered.
"""

import os
import dotenv
import pytest

# Neutralise dotenv before src modules import it, so a developer's real .env is ignored
dotenv.load_dotenv = lambda *args, **kwargs: False
dotenv.find_dotenv = lambda *args, **kwargs: ""

# Deterministic placeholders. No real key ever reaches a test.
os.environ.update({
    "TAVILY_API_KEY": "test-tavily-key",
    "BLOGGER_BLOG_ID": "test-blog-id",
    "WORKER_MODEL": "test-worker-model",
    "EDITOR_MODEL": "test-editor-model",
    "LANGSMITH_TRACING": "false",
})
os.environ.pop("POSTGRES_DB_URL", None)


class FakeResponse:
    """Stands in for a LangChain message returned by ChatOllama.invoke()."""

    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Records the prompt it was given and replays a canned response."""

    def __init__(self, content: str):
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return FakeResponse(self.content)


@pytest.fixture
def fake_llm():
    return FakeLLM
