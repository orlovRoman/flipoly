from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from polyflip.ai_lab.shadow import observation_key


def test_shadow_observation_key_is_idempotent_for_same_snapshot():
    snapshot = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert observation_key(3, "m1", snapshot) == observation_key(3, "m1", snapshot)
    assert observation_key(3, "m1", snapshot) != observation_key(3, "m1", snapshot + timedelta(seconds=1))


def test_shadow_payload_preserves_conflict_and_abstain_values():
    values = {
        "active_action": "ABSTAIN",
        "candidate_action": "BUY_YES",
        "lr_direction_vote": "DOWN",
        "lgbm_direction_vote": "UP",
        "consensus_type": "CONFLICT",
        "actual_combined_action": "ABSTAIN",
    }
    row = SimpleNamespace(**values)
    assert row.active_action == "ABSTAIN"
    assert row.candidate_action == "BUY_YES"
    assert row.consensus_type == "CONFLICT"
