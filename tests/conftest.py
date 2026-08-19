"""Keeps the suite hermetic: no .env, no credentials, no live model or API calls."""

import os
import dotenv
import pytest

# Neutralise dotenv before any src import, so a real .env is never read.
dotenv.load_dotenv = lambda *args, **kwargs: False
dotenv.find_dotenv = lambda *args, **kwargs: ""

# Placeholder keys, so no real credential reaches a test.
os.environ.update({
    "TAVILY_API_KEY": "test-tavily-key",
    "GEMINI_API_KEY": "test-gemini-key",  # the validator builds a client at import
    "BLOGGER_BLOG_ID": "test-blog-id",
    "WORKER_MODEL": "test-worker-model",
    "EDITOR_MODEL": "test-editor-model",
    "LANGSMITH_TRACING": "false",
})
os.environ.pop("POSTGRES_DB_URL", None)


class FakeResponse:
    """Stands in for a message returned by a chat model."""

    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Records the prompt it was given and replays a canned reply."""

    def __init__(self, content: str):
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return FakeResponse(self.content)


@pytest.fixture
def fake_llm():
    return FakeLLM
