from uris_platform.ui.components import metric_cards_html


def test_metric_cards_html_is_compact_and_does_not_emit_indented_block_markup():
    html = metric_cards_html(
        [
            {"label": "A", "value": "1", "sub": "first"},
            {"label": "B", "value": "2", "sub": "second"},
        ]
    )

    assert html.startswith('<div class="uris-grid"><div class="uris-card">')
    assert html.count('<div class="uris-card">') == 2
    assert "\n    <div class=\"uris-card\">" not in html


def test_metric_cards_html_escapes_values():
    html = metric_cards_html(
        [
            {"label": "<b>X</b>", "value": "<script>", "sub": "a&b"},
        ]
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&amp;" in html
