"""Scene-name sanitization + collision fallback for per-scene store dirnames."""

import pytest

from zarrmony.naming import resolve_scene_dirnames, sanitize_scene_name


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("scene1", "scene1"),
        ("Scene Name", "Scene_Name"),
        ("a/b:c", "a_b_c"),
        ("  trim_me  ", "trim_me"),
        ("a___b", "a___b"),
        ("a..b", "a..b"),
        (".hidden", "hidden"),
        ("__leading", "leading"),
        ("trailing__", "trailing"),
        ("with-dash_and.dot", "with-dash_and.dot"),
        ("01-01_pos1", "01-01_pos1"),
        ("ünicode!", "nicode"),
        ("", "scene"),
        ("///", "scene"),
        ("...", "scene"),
    ],
)
def test_sanitize_basic_cases(raw: str, expected: str) -> None:
    assert sanitize_scene_name(raw) == expected


def test_resolve_no_collisions_returns_unsuffixed() -> None:
    assert resolve_scene_dirnames(["alpha", "beta", "gamma"]) == [
        "alpha",
        "beta",
        "gamma",
    ]


def test_resolve_collision_suffixes_all_collisions() -> None:
    # Both "a/b" and "a:b" sanitize to "a_b" -> both get suffixed by index.
    assert resolve_scene_dirnames(["a/b", "a:b", "c"]) == ["a_b__0", "a_b__1", "c"]


def test_resolve_three_way_collision() -> None:
    assert resolve_scene_dirnames(["x y", "x_y", "x/y"]) == [
        "x_y__0",
        "x_y__1",
        "x_y__2",
    ]


def test_resolve_empty_names_collide_to_scene() -> None:
    assert resolve_scene_dirnames(["", "///", "ok"]) == ["scene__0", "scene__1", "ok"]


def test_resolve_preserves_index_order_independent_of_collision_position() -> None:
    # "alpha" appears at indices 0 and 2 — both should be suffixed with their
    # own index, not a counter.
    assert resolve_scene_dirnames(["alpha", "beta", "alpha"]) == [
        "alpha__0",
        "beta",
        "alpha__2",
    ]


def test_resolve_handles_empty_list() -> None:
    assert resolve_scene_dirnames([]) == []
