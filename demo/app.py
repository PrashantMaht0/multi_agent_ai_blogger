"""Replays recorded AI Blogger runs step by step. No models, no API keys, no publishing."""

import json
import time
from pathlib import Path

import gradio as gr

try:
    import spaces  # preinstalled on ZeroGPU Spaces, absent locally

    @spaces.GPU
    def _zero_gpu_placeholder():
        """Never called: the replay needs no GPU, but ZeroGPU will not start without one GPU function."""
except ImportError:
    pass

RUNS_DIR = Path(__file__).parent / "runs"
REPO_URL = "https://github.com/PrashantMaht0/multi_agent_ai_blogger"
ATTACK_SLUGS = {"injection"}
SPEEDS = {"20x": 20, "1x (real time)": 1, "Instant": 0}
MAX_PAUSE = 4.0  # seconds; keeps a replay watchable even at 1x


def load_runs() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*.json"))}


RUNS = load_runs()


def label(slug: str) -> str:
    run = RUNS[slug]
    prefix = "[Attack topic] " if slug in ATTACK_SLUGS else ""
    return f"{prefix}{run['topic']}"


def event_lines(node: str, update: dict) -> str:
    """The same trace lines the dashboard prints for each node."""
    if node == "planner":
        questions = update.get("sub_questions", [])
        lines = [f"[PLAN] {len(questions)} sub questions, timeframe {(update.get('timeframe') or {}).get('label', 'any time')}"]
        lines += [f"   {q['id']}: {q['question']}" for q in questions]
        return "\n".join(lines)
    if node == "research_branch":
        lines = [f"[BRANCH] {e['sub_question_id']} failed: {e['error']}" for e in update.get("branch_errors", [])]
        claims = update.get("claims", [])
        if claims:
            lines.append(f"[BRANCH] {claims[0]['sub_question_id']}: {len(claims)} claims")
        return "\n".join(lines)
    if node == "auditor":
        return f"[AUDIT] {update.get('audit_status')}: {update.get('audit_feedback')}"
    if node == "cross_check":
        gaps = update.get("gaps", [])
        important = sum(1 for g in gaps if g.get("important"))
        return f"[CROSS CHECK] {len(update.get('contradictions', []))} contradictions, {len(gaps)} gaps ({important} important)"
    if node == "gap_planner":
        return f"[GAP] researching {len(update.get('sub_questions', []))} more sub questions"
    if node == "writer":
        return "[WRITER] Draft written."
    if node == "editor":
        return f"[EDITOR] {update.get('last_evaluation')} (revision {update.get('revision_count')})"
    if node == "sanitizer":
        figures = update.get("unsupported_figures") or []
        return "[SANITIZER] " + (f"figures not in the research: {', '.join(figures)}" if figures else "clean")
    if node == "abort":
        return "[ABORT] Research did not pass the audit."
    return f"[{node.upper()}]"


def details(run: dict) -> str:
    """Plan, supported claims with quotes and sources, disagreements, review flag."""
    final = run["final"]
    verdicts = final.get("claim_verdicts") or {}
    questions = {q["id"]: q["question"] for q in final.get("sub_questions") or []}
    out = [f"**Timeframe:** {(final.get('timeframe') or {}).get('label', 'any time')}  ·  "
           f"**Run time:** {run['total_seconds'] / 60:.1f} min  ·  **Draft:** {run['draft_words']} words  ·  "
           f"**Audit:** {final.get('audit_feedback')}"]
    for qid, question in questions.items():
        supported = [c for c in final.get("claims") or []
                     if c["sub_question_id"] == qid and verdicts.get(c["id"], {}).get("status") == "supported"]
        out.append(f"\n**{qid}. {question}**")
        if not supported:
            out.append("- _no supported claims_")
        for c in supported:
            out.append(f"- {c['text']}  \n  > “{c['quote']}”  \n  [{c['source_title']}]({c['source_url']})")
    if final.get("contradictions"):
        out.append("\n**Sources disagree**")
        out += [f"- {c['note']}" for c in final["contradictions"]]
    if final.get("review_flag"):
        figures = final.get("unsupported_figures") or []
        out.append(f"\n**NEEDS REVIEW**" + (f": figures not in the research: {', '.join(figures)}" if figures else ""))
    return "\n".join(out)


def replay(slug: str, speed: str):
    run = RUNS[slug]
    factor = SPEEDS[speed]
    log = f"[TOPIC] {run['topic']}\n[RECORDED] {run['recorded_on']} on a 16 GB laptop, local qwen3 + Gemini auditor\n\n"
    draft = ""
    yield log, draft, ""
    previous = 0.0
    for event in run["events"]:
        if factor:
            time.sleep(min((event["t"] - previous) / factor, MAX_PAUSE))
        previous = event["t"]
        log += f"{event['t']:>6.0f}s  {event_lines(event['node'], event['update'])}\n"
        draft = event["update"].get("draft", draft)
        yield log, draft, ""
    log += f"\n[PAUSED] Awaiting human review. Total {run['total_seconds'] / 60:.1f} min."
    yield log, run["final"].get("draft") or "", details(run)


def _theme():
    try:
        return gr.Theme.from_hub("harsh8001/cartoon-style")
    except Exception:
        return gr.themes.Soft()


CSS = """
.gradio-container { padding: 20px 24px !important; max-width: 100% !important; }
.gradio-container { --radius-sm: 4px; --radius-md: 6px; --radius-lg: 8px; --radius-xl: 10px; --radius-xxl: 12px; }
.block { overflow: visible !important; }
"""

with gr.Blocks(title="AI Blogger - Run Replay") as demo:
    gr.Markdown(
        f"""# Multi-Agent Blogger Studio: run replay
Replays of **real, recorded runs** of the pipeline: a planner splits the topic, research branches pull
quoted claims from web pages, Gemini audits every claim against its quote, a cross check finds gaps and
disagreements, then the writer and editor produce the draft. Nothing here calls a model or an API.
Source and full pipeline: [{REPO_URL}]({REPO_URL})"""
    )
    with gr.Row():
        run_choice = gr.Dropdown(choices=[(label(s), s) for s in RUNS], value=next(iter(RUNS), None),
                                 label="Recorded run", scale=3)
        speed = gr.Radio(list(SPEEDS), value="20x", label="Replay speed", scale=2)
        play = gr.Button("Play", variant="primary", scale=1)
    with gr.Row():
        with gr.Column(scale=1):
            trace = gr.Textbox(label="Pipeline trace", lines=24, max_lines=40, autoscroll=True)
        with gr.Column(scale=1):
            draft = gr.HTML(label="Draft")
    research = gr.Markdown()
    play.click(replay, inputs=[run_choice, speed], outputs=[trace, draft, research])

if __name__ == "__main__":
    demo.launch(theme=_theme(), css=CSS)
