from pathlib import Path
import hashlib
import json

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "weighted_policy"
    / "fixture_30.json"
)


def test_real_market_legacy_fixture_is_resolved_and_stable():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    observations = payload["observations"]
    assert payload["fixture_size"] == 30
    assert len(observations) == 30
    assert len({item["market_id"] for item in observations}) == 30
    assert all(item["outcome_yes"] is not None for item in observations)

    legacy_lines = "\n".join(
        f"{item['market_id']}|{item['timestamp']}|{item.get('legacy_action') or 'SKIP'}"
        for item in observations
    )
    digest = hashlib.sha256(legacy_lines.encode("utf-8")).hexdigest()
    assert payload["legacy_decision_fingerprint"] == digest
    assert digest == "aa5f1708e64caed7c3a2b9e85de598d21480584f3476134f47f4f08bda64d59a"
