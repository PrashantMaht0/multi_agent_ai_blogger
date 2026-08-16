"""
src/prompts/__init__.py
Loads a versioned prompt definition from src/prompts/<name>.yaml.

Each agent loads only its own file. Templates use mustache-style {{variable}} markers so
literal JSON braces in a prompt need no escaping.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_ollama import ChatOllama

PROMPTS_DIR = Path(__file__).parent

# model: ${ENV_VAR:default}  -> env var if set, otherwise the default after the first colon
_ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+):(.*)\}$")
_VARIABLE_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _resolve_model(value: str) -> str:
    match = _ENV_PATTERN.match(value)
    return os.getenv(match.group(1), match.group(2)) if match else value


@dataclass
class Prompt:
    """A prompt template plus the model settings it was tuned against."""

    name: str
    version: str
    template: str
    model: str
    temperature: float = 0.0
    format: str | None = None
    num_predict: int | None = None
    reasoning: bool | None = None
    description: str = ""
    input_variables: list[str] = field(default_factory=list)

    def render(self, **values) -> str:
        """Substitutes {{variable}} markers. Raises if the template needs a value we lack."""
        missing = set(_VARIABLE_PATTERN.findall(self.template)) - set(values)
        if missing:
            raise KeyError(f"Prompt '{self.name}' is missing values for: {sorted(missing)}")
        return _VARIABLE_PATTERN.sub(lambda m: str(values[m.group(1)]), self.template)

    def llm(self, **overrides) -> ChatOllama:
        """Builds the chat model this prompt declares.

        Called per invocation, never cached at import: a ChatOllama holds an httpx client
        bound to the event loop that first used it, and asyncio.run() closes that loop.
        """
        settings = {
            "model": self.model,
            "temperature": self.temperature,
            "base_url": os.getenv("OLLAMA_BASE_URL"),
            # Hard ceiling on generation. A judge that should answer in 7 tokens has been
            # observed running to 2900 tokens of malformed JSON, which then fails to parse.
            "num_predict": self.num_predict,
            # gemma4:12b is a thinking model: given a long prompt it spends the whole
            # generation reasoning and returns empty content. Judges set this to false.
            "reasoning": self.reasoning,
        }
        if self.format:
            settings["format"] = self.format
        settings.update(overrides)
        return ChatOllama(**{k: v for k, v in settings.items() if v is not None})


def load_prompt(name: str) -> Prompt:
    """Reads src/prompts/<name>.yaml."""
    data = yaml.safe_load((PROMPTS_DIR / f"{name}.yaml").read_text())
    return Prompt(
        name=data["name"],
        version=str(data["version"]),
        template=data["template"],
        model=_resolve_model(data["model"]),
        temperature=data.get("temperature", 0.0),
        format=data.get("format"),
        num_predict=data.get("num_predict"),
        reasoning=data.get("reasoning"),
        description=data.get("description", ""),
        input_variables=data.get("input_variables", []),
    )
