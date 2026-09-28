"""Guards the prompt directory, with no model calls."""

import pytest
import yaml

from src.prompts import PROMPTS_DIR, Prompt, _resolve_model, load_prompt

AGENTS = ["planner", "researcher", "auditor", "cross_check", "writer", "editor"]


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
        "planner": {"topic": "MCP", "current_year": 2026},
        "researcher": {"question": "What is MCP?", "timeframe_label": "any time", "sources": "[1] URL: u"},
        "auditor": {"question": "What is MCP?", "claims": "[q1-1] CLAIM: x"},
        "cross_check": {"topic": "MCP", "timeframe_label": "any time", "sub_questions": "- a", "claims": "(none)"},
        "writer": {"topic": "MCP", "research": "- a fact", "feedback": "none", "target_words": 600},
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


def test_auditor_runs_on_a_hosted_model_with_current_knowledge():
    """Fact-checking needs a model with current knowledge."""
    auditor = load_prompt("auditor")

    assert auditor.model.startswith("gemini")
    assert auditor.num_predict


def test_only_the_auditor_and_cross_check_are_hosted():
    """The two fact-checking steps are hosted; every other agent runs locally."""
    hosted = [n for n in AGENTS if load_prompt(n).model.startswith("gemini")]
    assert hosted == ["auditor", "cross_check"]


def test_publish_has_no_prompt_or_model():
    """Publishing is a deterministic tool call, so no model can alter the approved draft."""
    import src.tools.publish as publish

    assert not (PROMPTS_DIR / "publisher.yaml").exists()
    assert not hasattr(publish, "prompt_spec")


def test_each_agent_loads_only_its_own_prompt():
    import src.agents.auditor as auditor
    import src.agents.editor as editor
    import src.agents.planner as planner
    import src.agents.research_branch as research_branch
    import src.agents.writer as writer

    for prompt, name in [(planner.prompt_spec, "planner"), (research_branch.prompt_spec, "researcher"),
                         (auditor.audit_prompt, "auditor"), (auditor.cross_check_prompt, "cross_check"),
                         (writer.prompt_spec, "writer"), (editor.prompt_spec, "editor")]:
        assert prompt.name == name


def test_llm_honours_ollama_base_url(monkeypatch):
    """Set in example.env and required once the app runs in a container."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    assert load_prompt("writer").llm().base_url == "http://host.docker.internal:11434"


# Regression: Ollama's default context silently cut extraction prompts to about 2,050 tokens.

QWEN3_PROMPTS = ["planner", "researcher", "writer"]


def test_extraction_context_fits_three_full_pages_plus_the_reply():
    """3 pages of 6,000 characters plus the template. Link heavy pages measured about 2.2 characters a token."""
    from src.mcp_servers.search_server import MAX_SOURCE_CHARS

    researcher = load_prompt("researcher")
    prompt_tokens = (3 * MAX_SOURCE_CHARS + len(researcher.template) + 600) / 2.2
    assert researcher.num_ctx >= prompt_tokens + researcher.num_predict


def test_qwen3_prompts_run_without_thinking():
    """Thinking made extraction 2.3x slower, and once used the writer's whole output cap (empty draft)."""
    assert all(load_prompt(n).reasoning is False for n in QWEN3_PROMPTS)


def test_qwen3_prompts_share_one_context_size():
    """A different num_ctx per call makes Ollama reload the model between steps."""
    sizes = {load_prompt(n).num_ctx for n in QWEN3_PROMPTS}
    assert len(sizes) == 1 and None not in sizes


def test_llm_passes_num_ctx_to_ollama():
    assert load_prompt("researcher").llm().num_ctx == load_prompt("researcher").num_ctx


def test_every_local_prompt_caps_its_output():
    """Regression: an uncapped qwen3 extraction looped past 56,000 tokens and hung a run for 80 minutes."""
    local = [n for n in AGENTS if not load_prompt(n).model.startswith("gemini")]
    uncapped = [n for n in local if not load_prompt(n).num_predict]
    assert local and uncapped == []


def test_extraction_cap_leaves_room_for_five_quoted_claims():
    assert 1500 <= load_prompt("researcher").num_predict < load_prompt("researcher").num_ctx
