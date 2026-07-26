import pytest
from polyflip.collector.resolver import extract_final_outcome

@pytest.mark.parametrize("market_data, expected", [
    ({"answer": "Yes"}, "YES"),
    ({"answer": "Up"}, "YES"),
    ({"winnerOutcome": "Down"}, "NO"),
    ({"resolvedBy": "0x123..."}, None),
    ({"outcomePrices": ["1", "0"], "outcomes": ["Yes", "No"]}, "YES"),
    ({"outcomePrices": ["0", "1"], "outcomes": ["Yes", "No"]}, "NO"),
    ({"outcomePrices": ["0.52", "0.48"], "outcomes": ["Yes", "No"]}, None),
    ({"answer": "INVALID"}, "INVALID"),
    ({"answer": "unknown value"}, None),
    ({"answer": "No"}, "NO"),
])
def test_extract_final_outcome(market_data, expected):
    assert extract_final_outcome(market_data) == expected
