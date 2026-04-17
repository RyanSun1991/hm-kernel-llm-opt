from hmopt.agents.prompting import render_prompt_template


def test_render_prompt_template_uses_fallback_and_preserves_unknown_placeholders():
    rendered = render_prompt_template(
        "nonexistent-template.md",
        "Hello {name} {missing}",
        name="kernel",
    )
    assert rendered == "Hello kernel {missing}"
