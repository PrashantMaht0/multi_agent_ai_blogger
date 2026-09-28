"""Researcher branch: one sub question into claims whose quotes are checked against their pages."""

import os
import re
import json
import threading

from src.common.errors import root_cause
from src.common.parsing import message_text, prompt_messages
from src.agents.planner import ANY_TIME
from src.prompts import load_prompt
from src.state import SubQuestion, Timeframe, _published, make_claims
from src.tools import search

prompt_spec = load_prompt("researcher")
researcher_llm = prompt_spec.llm()

# Shorter quotes match too many pages to prove anything.
MIN_QUOTE_CHARS = 20

# qwen3 reads only the best part of each page; the quote check still reads the whole page.
PAGE_BUDGET = 1500
PASSAGE_CHARS = 500
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = frozenset(
    "the and for are was were been being has have had does did doing what which who whom whose "
    "when where why how this that these those with from into onto over under about than then "
    "there their they them its it's can could would should will may might must also any all each "
    "more most some such only other own same very your you our not nor but just".split()
)


def question_words(question: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", question.lower()) if len(w) >= 3 and w not in STOPWORDS}


def _trim(text: str, limit: int) -> str:
    """Cuts at the last whole word within the limit."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return cut[:space] if space > 0 else cut


def _passages(text: str) -> list[str]:
    """Consecutive sentences joined up to about PASSAGE_CHARS; a longer sentence stands alone."""
    passages, current = [], ""
    for sentence in _SENTENCE_END.split(text.strip()):
        if current and len(current) + 1 + len(sentence) > PASSAGE_CHARS:
            passages.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        passages.append(current)
    return passages


def select_passages(text: str, question: str, budget: int = PAGE_BUDGET) -> str:
    """The page's passages that share the most words with the question, in page order."""
    if len(text) <= budget:
        return text
    words = question_words(question)
    passages = _passages(text)
    scores = [len(words & set(re.findall(r"[a-z]+", p.lower()))) for p in passages]
    if not any(scores):
        return _trim(text, budget)

    ranked = sorted(range(len(passages)), key=lambda i: (-scores[i], i))
    if len(passages[ranked[0]]) > budget:
        return _trim(passages[ranked[0]], budget)
    chosen, used = [], 0
    for i in ranked:
        cost = len(passages[i]) + (1 if chosen else 0)  # joined with a newline
        if used + cost <= budget:
            chosen.append(i)
            used += cost
    return "\n".join(passages[i] for i in sorted(chosen))


def _parallel_limit() -> int:
    try:
        return max(int(os.getenv("OLLAMA_NUM_PARALLEL", "")), 1)
    except ValueError:
        return 2


# Process wide on purpose: every branch, and every eval topic, shares one Ollama server.
_model_slots = threading.BoundedSemaphore(_parallel_limit())

_PUNCTUATION = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                              "–": "-", "—": "-"})


def normalize_text(text: str) -> str:
    """Straight quotes, plain dashes, collapsed whitespace, lowercase: the quote check's view."""
    return " ".join(text.translate(_PUNCTUATION).lower().split())


def _format_sources(sources: list[search.Source], question: str) -> str:
    return "\n\n".join(
        f"[{i}] URL: {s['url']}\nTITLE: {s['title']}\nDATE: {s.get('published_date') or 'unknown'}\n"
        f"TEXT:\n{select_passages(s['raw_content'], question)}"
        for i, s in enumerate(sources, 1)
    )


def _quoted_items(raw: list, sources: list[search.Source]) -> list[dict]:
    """Items whose quote is on one of the pages. Code, not the model, decides which page."""
    pages = [(s, normalize_text(s["raw_content"])) for s in sources]
    kept, seen = [], set()
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("quote"), str):
            continue
        needle = normalize_text(item["quote"])
        if len(needle) < MIN_QUOTE_CHARS:
            continue
        matches = [source for source, page in pages if needle in page]
        if not matches:
            continue
        # Keep the model's citation only when that page really holds the quote.
        source = next((m for m in matches if m["url"] == item.get("source_url")), matches[0])
        if (source["url"], needle) in seen:
            continue
        seen.add((source["url"], needle))
        kept.append({**item, "source_url": source["url"], "source_title": source["title"],
                     "published": _published(item.get("published")) or source.get("published_date")})
    return kept


def research_branch(sub_question: SubQuestion, timeframe: Timeframe | None) -> dict:
    """Returns claims, or exactly one branch error; never raises."""
    qid = sub_question["id"]

    def failed(reason: str) -> dict:
        print(f"[BRANCH] {qid} failed: {reason}")
        return {"claims": [], "branch_errors": [{"sub_question_id": qid, "error": reason}]}

    try:
        sources = search.search_sources(sub_question["question"])
    except Exception as e:
        return failed(root_cause(e))
    if not sources:
        return failed("no sources")

    prompt = prompt_spec.render(question=sub_question["question"],
                                timeframe_label=(timeframe or {}).get("label") or ANY_TIME,
                                sources=_format_sources(sources, sub_question["question"]))
    try:
        with _model_slots:
            reply = message_text(researcher_llm.invoke(prompt_messages(prompt, "Extract the claims now.")))
    except Exception as e:
        return failed(root_cause(e))

    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        return failed("unparseable reply")
    raw = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return failed("unparseable reply")

    claims = make_claims(qid, _quoted_items(raw, sources))
    if not claims:
        return failed("no quotable claims")
    print(f"[BRANCH] {qid}: {len(claims)} claims")
    return {"claims": claims, "branch_errors": []}
