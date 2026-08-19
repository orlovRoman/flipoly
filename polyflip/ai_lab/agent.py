"""Autonomous AI Lab Researcher Agent Core (Phase 10).

Executes end-to-end research iterations: hypothesis formation, training, OOT evaluation,
policy scoring, overlay management, and safe shadow assignment.
"""

from __future__ import annotations

import json
import time
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from polyflip.ai_lab.agent_tools import (
    apply_shadow_overlay,
    create_config_overlay,
    create_experiment_config_from_proposal,
    get_active_models,
    get_polymarket_oot_history,
    get_recent_trade_statistics,
)
from polyflip.ai_lab.llm import (
    AgentContext,
    AgentDecision,
    AnalysisContext,
    HypothesisProposal,
    LLMProvider,
    get_llm_provider,
)
from polyflip.ai_lab.lgbm_worker import execute_lgbm_steps
from polyflip.ai_lab.orchestrator import (
    evaluate_finalization_gate,
    evaluate_run,
    finalize_run,
    plan_run,
    promote_to_shadow,
)
from polyflip.ai_lab.policy import (
    evaluate_candidate_policy,
    validate_agent_action_autonomy,
)
from polyflip.ai_lab.service import (
    AILabError,
    append_step,
    propose_live_deployment,
    transition_run,
    utc_now,
)
from polyflip.db.models import (
    AIApprovalRequest,
    AIConfigOverlay,
    AIExperimentConfig,
    AIModelArtifact,
    AIOptimizationRun,
    AIRunStep,
    AIShadowAssignment,
    DeploymentRevision,
    ExperimentResult,
    ModelRegistry,
)

logger = structlog.get_logger("polyflip.ai_lab.agent")


class AILabAgent:
    """Autonomous quantitative researcher agent driving iterative optimization."""

    def __init__(
        self,
        session: AsyncSession,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.session = session
        self.llm = llm_provider or get_llm_provider()

    async def build_agent_context(self, run: AIOptimizationRun) -> AgentContext:
        """Construct a comprehensive context snapshot for LLM hypothesis generation."""
        scope = run.scope or {}
        asset = scope.get("asset", "BTCUSDT")

        # 1. Fetch active baseline models
        active_models = await get_active_models(self.session, asset)
        current_active = active_models[0] if active_models else None

        # 2. Fetch trade statistics
        stats = await get_recent_trade_statistics(self.session, asset, days=30)

        # 3. Fetch recent run steps for this optimization session
        recent_steps = (
            await self.session.execute(
                select(AIRunStep)
                .where(AIRunStep.run_id == run.id)
                .order_by(AIRunStep.id.desc())
                .limit(5)
            )
        ).scalars().all()

        prev_hypotheses = [
            {"step_id": s.id, "hypothesis": s.hypothesis, "summary": s.summary}
            for s in recent_steps
            if s.hypothesis
        ]

        budget = int(getattr(run, "budget_experiments", 0) or getattr(run, "experiment_budget", 0) or 0)
        remaining = max(0, budget - run.experiments_completed)

        return AgentContext(
            run_id=run.id,
            asset=asset,
            autonomy_level=run.autonomy_level or "EXPERIMENT",
            budget_remaining_steps=remaining,
            current_active_model=current_active,
            baseline_metrics={
                "accuracy": current_active.get("accuracy") if current_active else None,
                "backtest_pnl": current_active.get("backtest_pnl") if current_active else None,
            },
            feature_sets_available=["FS_D0", "FS_D1", "FS_D2", "FS_D3"],
            previous_hypotheses=prev_hypotheses,
            market_statistics=stats,
        )

    async def execute_iteration(self, run_id: int) -> dict[str, Any]:
        """Execute a single autonomous hypothesis -> train -> evaluate -> decide loop."""
        run = (
            await self.session.execute(
                select(AIOptimizationRun)
                .where(AIOptimizationRun.id == run_id)
                .with_for_update()
            )
        ).scalar_one_or_none()

        if run is None:
            raise AILabError(f"Run #{run_id} not found")

        if run.status in {"COMPLETED", "FAILED", "CANCELLED", "REJECTED", "PAUSED"}:
            return {"status": run.status, "message": f"Run is in terminal/paused state: {run.status}"}

        # Check budget availability
        if budget > 0 and run.experiments_completed >= budget:
            await transition_run(
                self.session,
                run,
                "COMPLETED",
                reason="Experiment budget exhausted",
            )
            return {"status": "COMPLETED", "message": "Experiment budget reached"}

        # 1. Build Context & Propose Hypothesis
        context = await self.build_agent_context(run)
        proposal, prop_stats = await self.llm.propose_hypothesis(context)

        step_idx = run.experiments_completed + 1
        existing_indices = (
            await self.session.execute(
                select(AIRunStep.step_index).where(AIRunStep.run_id == run.id)
            )
        ).scalars().all()
        next_step_index = max(existing_indices) + 1 if existing_indices else 0
        hypo_step = await append_step(
            self.session,
            run_id=run.id,
            step_index=next_step_index,
            step_type="HYPOTHESIS",
            hypothesis=proposal.hypothesis,
            action="PROPOSE_HYPOTHESIS",
            input_payload=proposal.model_dump(),
        )
        hypo_step.status = "SUCCEEDED"
        hypo_step.summary = f"LLM proposed hypothesis: {proposal.hypothesis[:150]}"
        await self.session.flush()

        # 2. Create Reproducible Config
        config = await create_experiment_config_from_proposal(self.session, proposal)

        # 3. Plan and Execute Model Training / Evaluation Steps
        planned_steps = await plan_run(
            self.session,
            run_id=run.id,
            config_ids=[config.id],
        )

        # 4. Execute the queued training and OOT steps in this dedicated worker.
        if planned_steps:
            await execute_lgbm_steps(
                self.session,
                run.id,
                max_steps=min(len(planned_steps), 10),
            )

        # 5. Evaluate Run results
        try:
            eval_report = await evaluate_run(self.session, run_id=run.id)
            metrics = eval_report.get("report", eval_report) if isinstance(eval_report, dict) else {}
        except Exception as exc:
            logger.info("evaluation_in_progress_or_no_results", run_id=run.id, error=str(exc))
            metrics = {
                "median_pnl": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
                "windows_count": 0,
            }

        # 6. Evaluate Quantitative Policy Rules
        policy_result = evaluate_candidate_policy(metrics)

        # 7. LLM Post-Experiment Analysis & Decision
        analysis_ctx = AnalysisContext(
            run_id=run.id,
            hypothesis=proposal,
            config_id=config.id,
            metrics=metrics,
            baseline_comparison={"score": policy_result.score},
            finalization_gate=policy_result.to_dict(),
            iteration=step_idx,
                budget_remaining_steps=max(0, budget - step_idx),
        )
        decision, dec_stats = await self.llm.analyze_experiment(analysis_ctx)

        # Record Analysis Step
        current_indices = (
            await self.session.execute(
                select(AIRunStep.step_index).where(AIRunStep.run_id == run.id)
            )
        ).scalars().all()
        analysis_step = await append_step(
            self.session,
            run_id=run.id,
            step_index=max(current_indices) + 1 if current_indices else next_step_index + 1,
            step_type="ANALYSIS",
            action="ANALYZE_EXPERIMENT",
            input_payload=decision.model_dump(),
        )
        analysis_step.status = "SUCCEEDED"
        analysis_step.summary = decision.rationale

        # 8. Apply Autonomy Decision & Gate Checks
        assigned_shadow = False
        applied_overlay = False
        proposed_live = False

        # Find candidate artifact if one was created
        artifact = (
            await self.session.execute(
                select(AIModelArtifact)
                .where(AIModelArtifact.config_id == config.id)
                .order_by(desc(AIModelArtifact.id))
                .limit(1)
            )
        ).scalar_one_or_none()

        if (
            policy_result.gate_passed
            and artifact
            and decision.action == "RECOMMEND_SHADOW"
        ):
            # Validate Autonomy level for SHADOW assignment
            try:
                validate_agent_action_autonomy("ASSIGN_SHADOW", run.autonomy_level)
                await promote_to_shadow(
                    self.session,
                    run_id=run.id,
                    candidate_artifact_id=artifact.id,
                    baseline_artifact_id=None,
                    asset=proposal.asset,
                )
                assigned_shadow = True
            except AILabError as e:
                logger.info("shadow_assignment_skipped_by_autonomy", reason=str(e))

        if (
            policy_result.gate_passed
            and artifact
            and decision.action == "REQUEST_LIVE_APPROVAL"
        ):
            # A live proposal is only an approval request, never activation.
            try:
                validate_agent_action_autonomy("PROPOSE_LIVE_DEPLOYMENT", run.autonomy_level)
                await propose_live_deployment(
                    self.session,
                    run_id=run.id,
                    actor="ai_agent",
                    reason=f"Candidate {artifact.id} cleared strict gate with score {policy_result.score:.4f}",
                )
                proposed_live = True
            except AILabError as e:
                logger.info("live_proposal_skipped_by_autonomy", reason=str(e))

        # Handle Configuration Overlay only when explicitly requested.
        if decision.proposed_overlay and decision.action == "APPLY_OVERLAY":
            try:
                validate_agent_action_autonomy("APPLY_CONFIG_OVERLAY", run.autonomy_level)
                overlay = await create_config_overlay(
                    self.session,
                    run_id=run.id,
                    changes=decision.proposed_overlay,
                    created_by="ai_agent",
                )
                await apply_shadow_overlay(self.session, overlay.id)
                applied_overlay = True
            except AILabError as e:
                logger.info("overlay_skipped_by_autonomy", reason=str(e))

        # 9. Update Run Counters & Summary
        run.experiments_completed += 1
        summary_payload = {
            "last_iteration": step_idx,
            "last_hypothesis": proposal.hypothesis,
            "decision": decision.action,
            "rationale": decision.rationale,
            "policy_score": policy_result.score,
            "gate_passed": policy_result.gate_passed,
            "assigned_shadow": assigned_shadow,
            "applied_overlay": applied_overlay,
            "proposed_live": proposed_live,
            "winner_artifact_id": artifact.id if artifact else None,
        }
        run.summary = json.dumps(summary_payload, ensure_ascii=False)

        if run.status == "EVALUATING":
            if run.experiments_completed >= budget:
                await transition_run(
                    self.session,
                    run,
                    "COMPLETED",
                    reason="experiment budget exhausted",
                )
            elif decision.action in {"FINALIZE_NO_WINNER", "STOP_BUDGET_EXHAUSTED"}:
                await transition_run(
                    self.session,
                    run,
                    "INSUFFICIENT_DATA",
                    reason="agent finalized without a policy winner",
                )
            else:
                await transition_run(
                    self.session,
                    run,
                    "QUEUED",
                    reason="agent queued the next research iteration",
                )

        await self.session.commit()
        return summary_payload
