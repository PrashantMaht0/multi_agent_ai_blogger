"""Loads a versioned prompt definition from src/prompts/<name>.yaml."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_ollama import ChatOllama

PROMPTS_DIR = Path(__file__).parent

# model: ${ENV_VAR:default} - env var if set, else the default after the first colon.
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
    repeat_penalty: float | None = None
    description: str = ""
    input_variables: list[str] = field(default_factory=list)

    def render(self, **values) -> str:
        """Substitutes {{variable}} markers, raising if a value is missing."""
        missing = set(_VARIABLE_PATTERN.findall(self.template)) - set(values)
        if missing:
            raise KeyError(f"Prompt '{self.name}' is missing values for: {sorted(missing)}")
        return _VARIABLE_PATTERN.sub(lambda m: str(values[m.group(1)]), self.template)

    def llm(self, **overrides):
        """Builds the chat model this prompt declares, hosted for gemini and local otherwise."""
        if self.model.startswith("gemini"):
            return self._google_llm(**overrides)

        settings = {
            "model": self.model,
            "temperature": self.temperature,
            "base_url": os.getenv("OLLAMA_BASE_URL"),
            "num_predict": self.num_predict,
            "reasoning": self.reasoning,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.format:
            settings["format"] = self.format
        settings.update(overrides)
        return ChatOllama(**{k: v for k, v in settings.items() if v is not None})

    def _google_llm(self, **overrides):
        """Builds a Gemini client, imported lazily so local-only runs need no key."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"Prompt '{self.name}' asks for {self.model}, but GEMINI_API_KEY is not set. "
                "Add it to your .env - see example.env."
            )

        settings = {
            "model": self.model,
            "google_api_key": api_key,
            "max_output_tokens": self.num_predict,
        }
        settings.update(overrides)
        return ChatGoogleGenerativeAI(**{k: v for k, v in settings.items() if v is not None})


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
        repeat_penalty=data.get("repeat_penalty"),
        description=data.get("description", ""),
        input_variables=data.get("input_variables", []),
    )
