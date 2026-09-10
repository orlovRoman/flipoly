import json

from polyflip.research.canonical_models.dev_compare import (
    ASK_MAX, ASK_MIN, PROVENANCE, _parse_ladder,
)


def test_ladder_parse_and_bounds():
    lad = _parse_ladder(json.dumps([{"price": 0.2, "size": 5}, {"price": 0.0, "size": 1}]))
    assert lad == [(0.2, 5.0)]
    assert _parse_ladder(None) == []
    assert (ASK_MIN, ASK_MAX) == (0.01, 0.40)


def test_provenance_label():
    assert PROVENANCE == "RECONSTRUCTED_CT"
