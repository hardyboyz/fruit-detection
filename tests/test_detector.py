"""Unit tests for label normalisation (no model needed)."""

from __future__ import annotations

import pytest

from app.detector import _iou, canonical_fruit, classify_label

CASES = [
    # raw label, expected display name (Indonesian), expected is_fruit
    ("apple", "Apel", True),
    ("Apple", "Apel", True),
    ("red_apple", "Apel", True),
    ("Granny Smith apple", "Apel", True),
    ("Ripe Banana", "Pisang Matang", True),
    ("raw-banana", "Pisang Mentah", True),
    ("Unripe Banana", "Pisang Mentah", True),
    ("Ripe Mango", "Mangga Matang", True),
    ("Raw Mango", "Mangga Mentah", True),
    ("Early-fruit Pomegranate", "Delima Muda", True),
    ("Mid-growth Pomegranate", "Delima Setengah Matang", True),
    ("Ripe Pomegranate", "Delima Matang", True),
    ("orange", "Jeruk", True),
    ("strawberry", "Stroberi", True),
    ("watermelon", "Semangka", True),
    ("pineapple", "Nanas", True),
    ("broccoli", "Brokoli", False),
    ("carrot", "Wortel", False),
]


@pytest.mark.parametrize("raw,expected,is_fruit", CASES)
def test_classify_label(raw, expected, is_fruit):
    display, category, fruit = classify_label(raw)
    assert display == expected
    assert fruit is is_fruit
    assert category == ("fruit" if is_fruit else "vegetable")


def test_ripeness_is_not_matched_mid_word():
    # "strawberry" contains "raw" but must not get a "Mentah" qualifier.
    assert classify_label("strawberry")[0] == "Stroberi"


def test_unknown_label_is_kept_and_treated_as_fruit():
    display, category, is_fruit = classify_label("some_new_class")
    assert display == "Some New Class"
    assert is_fruit is True
    assert category == "fruit"


def test_canonical_fruit():
    assert canonical_fruit("Granny Smith") == "Apple"
    assert canonical_fruit("Ripe Pomegranate") == "Pomegranate"
    assert canonical_fruit("spaghetti squash") is None


def test_iou():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert _iou((0, 0, 10, 10), (0, 0, 5, 10)) == pytest.approx(0.5)
