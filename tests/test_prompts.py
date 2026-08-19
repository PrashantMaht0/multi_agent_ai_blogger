"""Guards the prompt directory, with no model calls."""

import pytest
import yaml

from src.prompts import PROMPTS_DIR, Prompt, _resolve_model, load_prompt

AGENTS = ["researcher", "validator", "writer", "editor", "publisher"]


@pytest.mark.parametrize("name", AGENTS)
def test_every_agent_has_a_prompt_file(name):
    assert (PROMPTS_DIR / f"{name}.yaml").exists()


@pytest.mark.parametrize("name", AGENTS)
def test_prompt_declares_what_the_eval_workflow_needs(name):
    prompt = load_prompt(name)

    assert prompt.name == name
    assert prompt.version
    assert prompt.model
    assert prompt.template.strip()


def test_render_substitutes_variables_and_leaves_json_braces_alone():
    """Mustache markers leave literal braces in a prompt alone."""
    prompt = Prompt(
        name="t", version="0", model="m",
        template='Topic: {{topic}}\nReply as {"status": "PASS", "feedback": ""}',
    )

    rendered = prompt.render(topic="MCP")

    assert "Topic: MCP" in rendered
    assert '{"status": "PASS", "feedback": ""}' in rendered
    assert "{{" not in rendered


def test_agent_prompts_render_with_their_declared_variables():
    filled = {
        "validator": {"topic": "MCP", "research_notes": '["a fact"]', "error_context": ""},
        "writer": {"topic": "MCP", "research": "- a fact", "feedback": "none"},
        # The editor no longer receives research notes: it judges writing, not facts.
        "editor": {"topic": "MCP", "draft": "<p>x</p>"},
    }
    for name, values in filled.items():
        rendered = load_prompt(name).render(**values)
        assert "MCP" in rendered
        assert "{{" not in rendered


def test_render_refuses_to_silently_drop_a_variable():
    with pytest.raises(KeyError, match="feedback"):
        load_prompt("writer").render(topic="MCP", research="- fact")


def test_model_resolves_from_env_with_a_literal_fallback(monkeypatch):
    """An env var wins, and the fallback keeps a model name's own colon."""
    monkeypatch.setenv("WORKER_MODEL", "some-other-model")
    assert _resolve_model("${WORKER_MODEL:llama3:8b}") == "some-other-model"

    monkeypatch.delenv("WORKER_MODEL")
    assert _resolve_model("${WORKER_MODEL:llama3:8b}") == "llama3:8b"

    # A literal pinned in the YAML is used as-is
    assert _resolve_model("gemma4:12b") == "gemma4:12b"


def test_yaml_model_can_pin_a_model_for_an_ab_test(monkeypatch):
    monkeypatch.setenv("WORKER_MODEL", "ignored-when-pinned")
    assert load_prompt("writer").model == _resolve_model(
        yaml.safe_load((PROMPTS_DIR / "writer.yaml").read_text())["model"]
    )


def test_local_judge_disables_thinking_and_caps_output():
    """The local judge disables thinking and caps its output."""
    editor = load_prompt("editor")

    assert editor.reasoning is False
    assert editor.num_predict
    assert editor.format is None


def test_validator_runs_on_a_hosted_model_with_current_knowledge():
    """Fact-checking needs a model with current knowledge."""
    validator = load_prompt("validator")

    assert validator.model.startswith("gemini")
    assert validator.num_predict


def test_only_the_validator_is_hosted():
    """Only the validator is hosted; every other agent runs locally."""
    hosted = [n for n in AGENTS if load_prompt(n).model.startswith("gemini")]
    assert hosted == ["validator"]


def test_each_agent_loads_only_its_own_prompt():
    import src.agents.editor as editor
    import src.agents.publisher as publisher
    import src.agents.researcher as researcher
    import src.agents.validator as validator
    import src.agents.writer as writer

    for module, name in [(researcher, "researcher"), (validator, "validator"), (writer, "writer"),
                         (editor, "editor"), (publisher, "publisher")]:
        assert module.prompt_spec.name == name


def test_llm_honours_ollama_base_url(monkeypatch):
    """Set in example.env and required once the app runs in a container."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    assert load_prompt("writer").llm().base_url == "http://host.docker.internal:11434"
