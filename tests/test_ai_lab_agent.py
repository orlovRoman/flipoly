"""Comprehensive test suite for AI Lab Phase 10 (Autonomous Researcher Agent, Policy Engine & Overlays)."""

import asyncio
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

from polyflip.ai_lab.agent_tools import (
    ALLOWED_OVERLAY_KEYS,
    OVERLAY_BOUNDS,
    generate_feature_patch,
    validate_overlay_changes,
)
from polyflip.ai_lab.llm import (
    AgentContext,
    AgentDecision,
    AnalysisContext,
    HypothesisProposal,
    MockLLMProvider,
)
from polyflip.ai_lab.policy import (
    evaluate_candidate_policy,
    score_candidate,
    validate_agent_action_autonomy,
)
from polyflip.ai_lab.service import AILabError


class TestAILabAgentPhase10(unittest.TestCase):
    """Test suite covering Phase 10 schemas, policy engine, overlays, and autonomy guardrails."""

    def test_hypothesis_schema_valid(self):
        proposal = HypothesisProposal(
            hypothesis="Calibrated LogReg on FS_D1 improves outsider calibration for BTC",
            asset="BTC",
            market_role="FAVORITE",
            model_family="LOGREG",
            feature_set="FS_D1",
            parameter_changes={"C": 0.5},
            strategy_parameter_changes={"decision_threshold": 0.58},
            expected_effect={"metric": "median_oot_pnl", "direction": "increase"},
            reasoning=["Empirical stability on window 1"],
            risks=["Small volatility window"],
        )
        self.assertEqual(proposal.asset, "BTCUSDT")
        self.assertEqual(proposal.model_family, "LogisticRegression")
        self.assertEqual(proposal.feature_set, "FS_D1")

    def test_hypothesis_schema_rejects_disallowed_family(self):
        with self.assertRaises(ValidationError):
            HypothesisProposal(
                hypothesis="Arbitrary neural network optimization",
                asset="BTC",
                model_family="DeepNeuralNetwork",
                feature_set="FS_D0",
            )

    def test_hypothesis_schema_rejects_unsupported_asset(self):
        with self.assertRaises(ValidationError):
            HypothesisProposal(
                hypothesis="Test unlisted penny token",
                asset="ADA",
                model_family="LOGREG",
                feature_set="FS_D0",
            )

    def test_overlay_validation_allows_valid_keys(self):
        changes = {
            "MIN_EDGE": 0.035,
            "OUTSIDER_MAX_PRICE": 0.42,
            "DEAD_ZONE_WIDTH": 0.08,
            "FAVORITE_THRESHOLD": 0.60,
        }
        cleaned = validate_overlay_changes(changes)
        self.assertEqual(cleaned["MIN_EDGE"], 0.035)
        self.assertEqual(cleaned["OUTSIDER_MAX_PRICE"], 0.42)

    def test_overlay_validation_rejects_prohibited_keys(self):
        # Trying to modify database secrets or LIVE order gateways directly
        changes = {
            "MIN_EDGE": 0.03,
            "DATABASE_URL": "postgresql://hack:pass@evil/db",
        }
        with self.assertRaises(AILabError) as ctx:
            validate_overlay_changes(changes)
        self.assertIn("Prohibited overlay parameter", str(ctx.exception))

    def test_overlay_validation_rejects_out_of_bound_values(self):
        # MIN_EDGE cannot be set to unrealistic 50%
        changes = {"MIN_EDGE": 0.55}
        with self.assertRaises(AILabError) as ctx:
            validate_overlay_changes(changes)
        self.assertIn("violates safety bounds", str(ctx.exception))

    def test_policy_scoring_deterministic(self):
        score_a = score_candidate(
            median_pnl=5.25,
            max_drawdown=8.0,
            trade_count=80,
            windows_count=3,
            auc=0.62,
        )
        score_b = score_candidate(
            median_pnl=5.25,
            max_drawdown=8.0,
            trade_count=80,
            windows_count=3,
            auc=0.62,
        )
        self.assertEqual(score_a, score_b)
        self.assertGreater(score_a, 5.0)

    def test_policy_engine_rejection_criteria(self):
        # Candidate with negative PnL must be rejected
        metrics_bad_pnl = {
            "median_pnl": -1.2,
            "max_drawdown": 5.0,
            "total_trades": 60,
            "windows_count": 3,
        }
        res = evaluate_candidate_policy(metrics_bad_pnl)
        self.assertFalse(res.gate_passed)
        self.assertTrue(any("NON_POSITIVE_PNL" in r for r in res.rejection_reasons))

        # Candidate with < 50 trades must be rejected
        metrics_low_trades = {
            "median_pnl": 2.5,
            "max_drawdown": 4.0,
            "total_trades": 35,
            "windows_count": 3,
        }
        res_low = evaluate_candidate_policy(metrics_low_trades)
        self.assertFalse(res_low.gate_passed)
        self.assertTrue(any("INSUFFICIENT_TRADES" in r for r in res_low.rejection_reasons))

    def test_policy_engine_accepts_clean_candidate(self):
        metrics_good = {
            "median_pnl": 6.4,
            "max_drawdown": 7.5,
            "total_trades": 85,
            "windows_count": 3,
            "auc": 0.65,
        }
        res = evaluate_candidate_policy(metrics_good)
        self.assertTrue(res.gate_passed)
        self.assertTrue(res.is_eligible)
        self.assertEqual(len(res.rejection_reasons), 0)

    def test_autonomy_level_boundaries(self):
        # OBSERVE cannot train models
        with self.assertRaises(AILabError):
            validate_agent_action_autonomy("TRAIN_MODEL", "OBSERVE")

        # EXPERIMENT cannot assign SHADOW
        with self.assertRaises(AILabError):
            validate_agent_action_autonomy("ASSIGN_SHADOW", "EXPERIMENT")

        # AUTONOMOUS_SHADOW can assign SHADOW
        validate_agent_action_autonomy("ASSIGN_SHADOW", "AUTONOMOUS_SHADOW")

        # LIVE_PROPOSE cannot activate LIVE directly
        with self.assertRaises(AILabError):
            validate_agent_action_autonomy("ACTIVATE_LIVE_DEPLOYMENT", "LIVE_PROPOSE")

    def test_mock_llm_propose_and_analyze(self):
        async def run_async():
            provider = MockLLMProvider()
            ctx = AgentContext(
                run_id=10,
                asset="BTC",
                autonomy_level="EXPERIMENT",
                budget_remaining_steps=3,
            )
            proposal, pstats = await provider.propose_hypothesis(ctx)
            self.assertIn("BTC", proposal.asset)
            self.assertGreater(pstats.total_tokens, 0)

            actx = AnalysisContext(
                run_id=10,
                hypothesis=proposal,
                config_id=1,
                metrics={"median_pnl": 4.5},
                baseline_comparison={},
                finalization_gate={"passed": True},
                iteration=1,
                budget_remaining_steps=2,
            )
            decision, dstats = await provider.analyze_experiment(actx)
            self.assertEqual(decision.action, "RECOMMEND_SHADOW")

        asyncio.run(run_async())

    def test_feature_patch_generator(self):
        patch = generate_feature_patch("momentum_zscore", "zscore(close, 20)", "20-period momentum zscore")
        self.assertEqual(patch["feature_name"], "momentum_zscore")
        self.assertEqual(len(patch["patch_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
