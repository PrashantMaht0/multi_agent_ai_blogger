"""
src/agents/sanitize.py
Strips unsafe markup from a draft before it can be published.

The evaluation baseline proved an injected <script> tag survives the writer and the
editor and reaches the draft, and that whether it survives is chance rather than a
reliable behaviour. Publishing sends raw HTML to Blogger, so this runs as code, not as
another model judgement: a regex cannot be argued out of its decision by the content it
is inspecting.
"""

import re

# Tags removed with everything between their opening and closing form.
_DANGEROUS_BLOCKS = ("script", "iframe", "object", "embed", "style", "form", "svg")
# Tags removed on sight; they have no closing pair to worry about.
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
# Anything the model emits before the first real tag: a leaked chat role such as
# "assistant", a markdown fence, or a "Here is the post:" preamble. llama3.1:8b leaks the
# role token; qwen3 did not, so this is model-dependent and belongs in code.
_PREAMBLE_PATTERN = re.compile(r"^[^<]*?(?=<\s*[a-zA-Z])", re.DOTALL)
_TRAILING_FENCE_PATTERN = re.compile(r"```\s*$")


def sanitize_html(draft: str) -> tuple[str, list[str]]:
    """Returns the cleaned draft and a list of what was removed."""
    if not draft:
        return draft, []

    removed = []

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
