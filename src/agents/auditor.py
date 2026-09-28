"""Auditor, cross check and gap planner: decide which claims reach the writer."""

import json

from src.common.errors import root_cause
from src.common.parsing import message_text, prompt_messages, strip_json_fence
from src.agents.planner import ANY_TIME, normalize_question
from src.prompts import load_prompt
from src.state import AgentState, Claim, Gap, Verdict, render_notes

audit_prompt = load_prompt("auditor")
auditor_llm = audit_prompt.llm()
cross_check_prompt = load_prompt("cross_check")
cross_check_llm = cross_check_prompt.llm()

MIN_SUPPORTED_CLAIMS = 5
MIN_COVERED_QUESTIONS = 3
MAX_AUDIT_LOOPS = 2
MAX_GAP_QUESTIONS = 3


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _ask_json(llm, prompt: str, ask: str, is_valid) -> dict:
    """One call, retried once on an error or an invalid reply; raises if both fail."""
    reason = ""
    for _ in range(2):
        try:
            data = json.loads(strip_json_fence(message_text(llm.invoke(prompt_messages(prompt, ask)))))
            if is_valid(data):
                return data
            reason = "unparseable reply"
        except Exception as e:
            reason = root_cause(e)
    raise RuntimeError(reason)


def _off_timeframe(claim: Claim, timeframe) -> bool:
    if not timeframe or not claim.get("published"):
        return False
    year = int(claim["published"][:4])
    start, end = timeframe.get("start_year"), timeframe.get("end_year")
    return (start is not None and year < start) or (end is not None and year > end)


def _judge(question: str, claims: list[Claim]) -> dict[str, Verdict]:
    listing = "\n".join(f"[{c['id']}] CLAIM: {_one_line(c['text'])}\n  QUOTE: {_one_line(c['quote'])}\n"
                        f"  SOURCE: {_one_line(c['source_title'])}" for c in claims)
    data = _ask_json(auditor_llm, audit_prompt.render(question=question, claims=listing),
                     "Give your verdicts now.",
                     lambda d: isinstance(d, dict) and isinstance(d.get("verdicts"), list))
    ids = {c["id"] for c in claims}
    verdicts = {}
    for v in data["verdicts"]:
        if (isinstance(v, dict) and v.get("claim_id") in ids
                and v.get("status") in ("supported", "unsupported")):
            verdicts[v["claim_id"]] = {"status": v["status"], "reason": _one_line(v.get("reason") or "")}
    return verdicts


def _supported(state: AgentState, verdicts: dict[str, Verdict]) -> list[Claim]:
    return [c for c in state.get("claims", []) if verdicts.get(c["id"], {}).get("status") == "supported"]


def auditor_node(state: AgentState) -> dict:
    claims = state.get("claims", [])
    sub_questions = state.get("sub_questions", [])
    timeframe = state.get("timeframe")
    judged = state.get("claim_verdicts", {})
    pending = [c for c in claims if c["id"] not in judged]

    order = [q["id"] for q in sub_questions]
    order += sorted({c["sub_question_id"] for c in pending} - set(order))
    questions = {q["id"]: q["question"] for q in sub_questions}

    new_verdicts: dict[str, Verdict] = {}
    errors = []
    for qid in order:
        group = [c for c in pending if c["sub_question_id"] == qid]
        to_judge = []
        for claim in group:
            if _off_timeframe(claim, timeframe):
                new_verdicts[claim["id"]] = {"status": "off_timeframe",
                                             "reason": f"published {claim['published']}, outside {timeframe['label']}"}
            else:
                to_judge.append(claim)
        if not to_judge:
            continue
        try:
            new_verdicts.update(_judge(questions.get(qid, ""), to_judge))
        except Exception as e:
            errors.append({"sub_question_id": qid, "error": f"audit failed: {e}"})

    merged = {**judged, **new_verdicts}
    supported = _supported(state, merged)
    covered = len({c["sub_question_id"] for c in supported})
    passed = len(supported) >= MIN_SUPPORTED_CLAIMS and covered >= MIN_COVERED_QUESTIONS
    feedback = f"{len(supported)} supported claims across {covered} sub questions"
    if not passed:
        feedback += f"; need {MIN_SUPPORTED_CLAIMS} across {MIN_COVERED_QUESTIONS}"

    statuses = [merged.get(c["id"], {}).get("status") for c in claims]
    print(f"[AUDIT] {statuses.count('supported')} supported, {statuses.count('unsupported')} unsupported, "
          f"{statuses.count('off_timeframe')} off timeframe, {statuses.count(None)} unaudited: "
          f"{'PASSED' if passed else 'FAILED'}")
    return {
        "claim_verdicts": new_verdicts,
        "branch_errors": errors,
        "audit_status": "PASSED" if passed else "FAILED",
        "audit_feedback": feedback,
        "audit_loops": state.get("audit_loops", 0) + 1,
        "research_notes": render_notes(sub_questions, claims, merged, state.get("contradictions", [])),
        "sender": "auditor",
    }


def _valid_cross_check(data) -> bool:
    return (isinstance(data, dict) and isinstance(data.get("contradictions", []), list)
            and isinstance(data.get("gaps", []), list))


def cross_check_node(state: AgentState) -> dict:
    verdicts = state.get("claim_verdicts", {})
    supported = _supported(state, verdicts)
    supported_ids = {c["id"] for c in supported}
    timeframe = state.get("timeframe") or {}
    prompt = cross_check_prompt.render(
        topic=state["topic"],
        timeframe_label=timeframe.get("label") or ANY_TIME,
        sub_questions="\n".join(f"- {_one_line(q['question'])}" for q in state.get("sub_questions", [])),
        claims="\n".join(f"[{c['id']}] {_one_line(c['text'])} (Source: {_one_line(c['source_title'])})"
                         for c in supported) or "(none)",
    )
    try:
        data = _ask_json(cross_check_llm, prompt, "Give your review now.", _valid_cross_check)
    except Exception as e:
        print(f"[CROSS CHECK] failed, continuing without it: {e}")
        data = {}

    contradictions = []
    for item in data.get("contradictions", []):
        if not isinstance(item, dict) or not isinstance(item.get("claim_ids"), list):
            continue
        ids = [i for i in item["claim_ids"] if isinstance(i, str)]
        note = _one_line(item.get("note") or "")
        if note and len(set(ids)) >= 2 and len(ids) == len(item["claim_ids"]) and set(ids) <= supported_ids:
            contradictions.append({"claim_ids": ids, "note": note})

    gaps: list[Gap] = []
    for item in data.get("gaps", []):
        if isinstance(item, dict) and isinstance(item.get("question"), str) and item["question"].strip():
            gaps.append({"question": _one_line(item["question"]), "reason": _one_line(item.get("reason") or ""),
                         "important": item.get("important") is True})
    gaps = gaps[:MAX_GAP_QUESTIONS]

    print(f"[CROSS CHECK] {len(contradictions)} contradictions, {len(gaps)} gaps "
          f"({sum(g['important'] for g in gaps)} important)")
    return {
        "contradictions": contradictions,
        "gaps": gaps,
        "research_notes": render_notes(state.get("sub_questions", []), state.get("claims", []),
                                       verdicts, contradictions),
        "sender": "cross_check",
    }


def new_gaps(state: AgentState) -> list[Gap]:
    """Gaps that repeat no researched question and no earlier gap, at most 3."""
    seen = {normalize_question(q["question"]) for q in state.get("sub_questions", [])}
    fresh = []
    for gap in state.get("gaps", []):
        key = normalize_question(gap["question"])
        if key and key not in seen:
            seen.add(key)
            fresh.append(gap)
    return fresh[:MAX_GAP_QUESTIONS]


def gap_planner_node(state: AgentState) -> dict:
    highest = max((int(q["id"][1:]) for q in state.get("sub_questions", [])), default=0)
    added = [{"id": f"q{highest + i}", "question": g["question"], "origin": "gap"}
             for i, g in enumerate(new_gaps(state), 1)]
    print(f"[GAP] researching {len(added)} more sub questions")
    return {"sub_questions": added, "sender": "gap_planner"}
