"""Strips unsafe markup from a draft before it can be published."""

import html
import re

import nh3

# The writer's own allowlist; anything else is removed.
ALLOWED_TAGS = {"h2", "h3", "p", "strong", "em", "ul", "ol", "li", "code", "pre"}

_TAG_NAME_PATTERN = re.compile(r"<\s*/?\s*([a-zA-Z][\w:-]*)")
_PREAMBLE_PATTERN = re.compile(r"^[^<]*?(?=<\s*[a-zA-Z])", re.DOTALL)
_TRAILING_FENCE_PATTERN = re.compile(r"```\s*$")
MAX_TITLE_LENGTH = 150

_FIGURE_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
_CODE_BLOCK_PATTERN = re.compile(r"<(code|pre)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_URL_PATTERN = re.compile(r"https?://\S+")
_ANY_TAG_PATTERN = re.compile(r"<[^>]+>")


def sanitize_html(draft: str) -> tuple[str, list[str]]:
    """Returns the cleaned draft and a list of what was removed."""
    if not draft:
        return draft, []

    removed = []

    # Drop any role label, markdown fence or preamble before the first tag.
    cleaned = _TRAILING_FENCE_PATTERN.sub("", draft)
    stripped_preamble = _PREAMBLE_PATTERN.match(cleaned)
    if stripped_preamble and stripped_preamble.group().strip():
        removed.append(f"preamble before the first tag ({stripped_preamble.group().strip()[:40]!r})")
        cleaned = cleaned[stripped_preamble.end():]

    disallowed = {t.lower() for t in _TAG_NAME_PATTERN.findall(cleaned)} - ALLOWED_TAGS

    # Allowlist parse: only the listed tags survive, and no attributes at all.
    safe = nh3.clean(cleaned, tags=ALLOWED_TAGS, attributes={}, link_rel=None)

    if disallowed:
        removed.append(f"disallowed tag(s): {', '.join(sorted(disallowed))}")
    elif safe != cleaned:
        removed.append("attributes or malformed markup")

    return safe, removed


def _figures(text: str) -> list[str]:
    """Years, numbers of 10 or more, decimals and percentages; commas dropped, first seen order."""
    found = []
    for match in _FIGURE_PATTERN.findall(text):
        figure = match.replace(",", "")
        number = figure.rstrip("%")
        if not number:
            continue
        if (len(number) == 4 and number.isdigit()) or "." in number or figure.endswith("%") or float(number) >= 10:
            if figure not in found:
                found.append(figure)
    return found


def unsupported_figures(draft_html: str, notes: list[str]) -> list[str]:
    """Figures in the draft's text that no research note contains. Code and URLs are ignored."""
    text = _CODE_BLOCK_PATTERN.sub(" ", draft_html or "")
    text = html.unescape(_ANY_TAG_PATTERN.sub(" ", _URL_PATTERN.sub(" ", text)))
    known = set(_figures(" ".join(notes)))
    return [f for f in _figures(text) if f not in known]


def clean_title(topic: str) -> str:
    """Turns the raw topic into a plain-text post title with no markup."""
    text = html.unescape(nh3.clean(topic, tags=set()))
    text = re.sub(r"\s+", " ", text.replace("<", "").replace(">", "")).strip()
    return text[:MAX_TITLE_LENGTH]
