"""Strips unsafe markup from a draft before it can be published."""

import re

_DANGEROUS_BLOCKS = ("script", "iframe", "object", "embed", "style", "form", "svg")
_DANGEROUS_VOID = ("img", "input", "link", "meta", "base")

_BLOCK_PATTERN = re.compile(
    r"<\s*(%s)\b[^>]*>.*?<\s*/\s*\1\s*>" % "|".join(_DANGEROUS_BLOCKS),
    re.IGNORECASE | re.DOTALL,
)
_ORPHAN_TAG_PATTERN = re.compile(
    r"<\s*/?\s*(%s)\b[^>]*>" % "|".join(_DANGEROUS_BLOCKS + _DANGEROUS_VOID),
    re.IGNORECASE,
)
_EVENT_HANDLER_PATTERN = re.compile(r"\son[a-z]+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", re.IGNORECASE)
_JS_URL_PATTERN = re.compile(r"(href|src)\s*=\s*(\"|')?\s*javascript:[^\"'>\s]*(\"|')?", re.IGNORECASE)
_PREAMBLE_PATTERN = re.compile(r"^[^<]*?(?=<\s*[a-zA-Z])", re.DOTALL)
_TRAILING_FENCE_PATTERN = re.compile(r"```\s*$")


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

    cleaned, count = _BLOCK_PATTERN.subn("", cleaned)
    if count:
        removed.append(f"{count} script/iframe/style block(s)")

    cleaned, count = _ORPHAN_TAG_PATTERN.subn("", cleaned)
    if count:
        removed.append(f"{count} unsafe tag(s)")

    cleaned, count = _EVENT_HANDLER_PATTERN.subn("", cleaned)
    if count:
        removed.append(f"{count} inline event handler(s)")

    cleaned, count = _JS_URL_PATTERN.subn("", cleaned)
    if count:
        removed.append(f"{count} javascript: URL(s)")

    return cleaned, removed
