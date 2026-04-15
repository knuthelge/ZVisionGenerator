"""Tests for prompt loading — filtering and negative prompt extraction."""

from __future__ import annotations

import textwrap

from zvisiongenerator.utils.prompts import load_prompts_file


def test_load_prompts_filters_inactive(tmp_path):
    """Inactive entries (active: false) are excluded."""
    yaml_content = textwrap.dedent("""\
        my_set:
          - prompt: "active prompt"
          - prompt: "inactive prompt"
            active: false
          - prompt: "another active"
    """)
    f = tmp_path / "prompts.yaml"
    f.write_text(yaml_content)
    result = load_prompts_file(str(f))
    assert len(result["my_set"]) == 2


def test_load_prompts_extracts_negative(tmp_path):
    """Negative prompts are extracted as tuple second element."""
    yaml_content = textwrap.dedent("""\
        set1:
          - prompt: "a cat"
            negative: "blurry"
          - prompt: "a dog"
    """)
    f = tmp_path / "prompts.yaml"
    f.write_text(yaml_content)
    result = load_prompts_file(str(f))
    assert result["set1"][0] == ("a cat", "blurry")
    assert result["set1"][1] == ("a dog", None)


def test_snippets_list_raises(tmp_path):
    """A snippets block that is a list instead of a dict raises ValueError."""
    yaml_content = textwrap.dedent("""\
        snippets:
          - item1
          - item2
        my_set:
          - prompt: "hello"
    """)
    f = tmp_path / "prompts.yaml"
    f.write_text(yaml_content)
    import pytest

    with pytest.raises(ValueError, match="snippets.*must be a mapping"):
        load_prompts_file(str(f))


def test_snippets_scalar_raises(tmp_path):
    """A snippets block that is a scalar raises ValueError."""
    yaml_content = textwrap.dedent("""\
        snippets: "not a dict"
        my_set:
          - prompt: "hello"
    """)
    f = tmp_path / "prompts.yaml"
    f.write_text(yaml_content)
    import pytest

    with pytest.raises(ValueError, match="snippets.*must be a mapping"):
        load_prompts_file(str(f))


def test_null_prompt_raises(tmp_path):
    """prompt: null should be rejected after snippet resolution."""
    yaml_content = textwrap.dedent("""\
        my_set:
          - prompt: null
    """)
    f = tmp_path / "prompts.yaml"
    f.write_text(yaml_content)
    import pytest

    with pytest.raises(ValueError, match="empty prompt"):
        load_prompts_file(str(f))


def test_empty_string_prompt_raises(tmp_path):
    """prompt: '' should be rejected after snippet resolution."""
    yaml_content = textwrap.dedent("""\
        my_set:
          - prompt: ""
    """)
    f = tmp_path / "prompts.yaml"
    f.write_text(yaml_content)
    import pytest

    with pytest.raises(ValueError, match="empty prompt"):
        load_prompts_file(str(f))


def test_malformed_yaml_prompts_raises_valueerror(tmp_path):
    """Malformed YAML in a prompts file must raise ValueError."""
    bad = tmp_path / "bad_prompts.yaml"
    bad.write_text("prompts:\n  - prompt: hello\n  bad_indent", encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="Failed to parse prompts file"):
        load_prompts_file(str(bad))
