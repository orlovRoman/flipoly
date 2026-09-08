"""
tests/models/test_stage2_evaluation.py

Verification and regression tests for Stage 2 evaluation protocols,
paired block bootstrap, candidate bundling, and decision rules.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from scripts.research.run_stage2_price_filter_evaluation import (
    compute_paired_block_bootstrap,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_paired_block_bootstrap_identical():
    """Identical strategies must yield exactly 0 delta and 0 bounds."""
    pnls_a = [1.0, -0.5, 0.8, -0.2]
    pnls_b = [1.0, -0.5, 0.8, -0.2]
    clusters = ["m1", "m2", "m3", "m4"]
    res = compute_paired_block_bootstrap(pnls_a, pnls_b, clusters, n_bootstrap=500, seed=42)
    assert res["status"] == "VALID"
    assert res["delta_pnl"] == pytest.approx(0.0)
    assert res["ci_lower"] == pytest.approx(0.0)
    assert res["ci_upper"] == pytest.approx(0.0)


def test_paired_block_bootstrap_insufficient_blocks():
    """Single cluster should return INSUFFICIENT_BLOCKS status."""
    pnls_a = [1.0, 2.0]
    pnls_b = [0.0, 0.0]
    clusters = ["single_market", "single_market"]
    res = compute_paired_block_bootstrap(pnls_a, pnls_b, clusters)
    assert res["status"] == "INSUFFICIENT_BLOCKS"


def test_paired_block_bootstrap_strictly_positive():
    """Strong positive advantage across independent clusters yields lower CI > 0."""
    pnls_a = [10.0] * 20
    pnls_b = [-5.0] * 20
    clusters = [f"market_{i}" for i in range(20)]
    res = compute_paired_block_bootstrap(pnls_a, pnls_b, clusters, n_bootstrap=500, seed=42)
    assert res["status"] == "VALID"
    assert res["delta_pnl"] == pytest.approx(300.0)
    assert res["ci_lower"] > 0


def test_stage2_protocol_artifact():
    """Verify that stage2_evaluation_protocol.json is pre-registered and valid."""
    proto_path = REPO_ROOT / "artifacts" / "research" / "stage2_evaluation_protocol.json"
    assert proto_path.exists(), "stage2_evaluation_protocol.json missing!"
    with open(proto_path, "r", encoding="utf-8") as f:
        proto = json.load(f)

    assert "dataset_sha256" in proto
    assert "splits" in proto
    assert "development" in proto["splits"]
    assert "test_holdout" in proto["splits"]
    assert proto["splits"]["development"]["n_rows"] > 0
    assert proto["splits"]["test_holdout"]["n_rows"] > 0
    assert proto["policy_parameters"]["stake_usdc"] == 1.0
    assert proto["policy_parameters"]["min_edge"] == 0.02
    assert "0.40" in proto["selection_criteria"]["hypothesis"]


def test_stage2_verdict_artifact():
    """Verify that stage2_price_filter_verdict.json records honest non-zero-crossing verdict."""
    verdict_path = REPO_ROOT / "artifacts" / "research" / "stage2_price_filter_verdict.json"
    assert verdict_path.exists(), "stage2_price_filter_verdict.json missing!"
    with open(verdict_path, "r", encoding="utf-8") as f:
        verdict = json.load(f)

    assert "development_results" in verdict
    assert "development_robustness" in verdict
    assert "out_of_sample_test_results" in verdict
    assert "final_verdict" in verdict

    status = verdict["final_verdict"]["status"]
    # With CI crossing zero on test holdout, the status MUST be INCONCLUSIVE, not fake CANDIDATE_SELECTED
    assert status == "INCONCLUSIVE", f"Expected INCONCLUSIVE due to CI crossing zero, got {status}"


def test_candidate_bundle_artifact():
    """Verify candidate bundle configuration."""
    bundle_path = REPO_ROOT / "artifacts" / "research" / "candidate_bundle.json"
    assert bundle_path.exists(), "candidate_bundle.json missing!"
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    assert bundle["feature_set"] == "MODEL_A1"
    assert bundle["execution_policy"]["max_price"] == 0.40
    assert bundle["execution_policy"]["min_edge"] == 0.02
    assert "hyperparameters" in bundle


def test_development_robustness_grid_monotonicity():
    """Trade count must be monotonically non-decreasing as price cap increases."""
    verdict_path = REPO_ROOT / "artifacts" / "research" / "stage2_price_filter_verdict.json"
    with open(verdict_path, "r", encoding="utf-8") as f:
        verdict = json.load(f)

    caps_grid = verdict["development_robustness"]["caps_sensitivity"]
    caps_sorted = sorted(caps_grid.keys(), key=lambda k: float(k))
    trade_counts = [caps_grid[c]["n_trades"] for c in caps_sorted]

    for i in range(len(trade_counts) - 1):
        assert trade_counts[i] <= trade_counts[i + 1], (
            f"Trade counts non-monotonic: {trade_counts[i]} > {trade_counts[i+1]} at cap {caps_sorted[i]}"
        )
