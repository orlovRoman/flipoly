"""Autonomous research loop for the independent AI research agent.

One iteration: claim -> context -> hypothesis -> proposal -> wait TRAIN/OOT ->
decision -> complete. All side effects go through the AI Lab HTTP API.
"""
from __future__ import annotations

import asyncio
import logging
import signal

from api_client import AILabApiClient, AgentAPIError, LeaseLostError
from opencode_client import OpenCodeClient
from schemas import ClaimedRun

logger = logging.getLogger("ai_research_agent")

POLL_SECONDS = float(__import__("os").getenv("AI_LAB_POLL_SECONDS", "10"))
IDLE_SLEEP_SECONDS = max(POLL_SECONDS, 2.0)


def _context_for_llm(run: ClaimedRun, context) -> dict:
    return {
        "run_id": run.id,
        "objective": run.objective,
        "scope": run.scope,
        "autonomy_level": run.autonomy_level,
        "budget_remaining_steps": max(
            run.budget_experiments - run.experiments_completed, 0
        ),
        "research_model": run.llm_research_model or "",
        "summary_model": run.llm_summary_model or "",
        "active_models": context.active_models,
        "recent_trade_statistics": context.recent_trade_statistics,
        "prior_experiments": context.prior_experiments,
        "available_feature_sets": context.available_feature_sets,
        "quality_gate": context.quality_gate,
    }


async def process_one_run(client: AILabApiClient, llm: OpenCodeClient) -> bool:
    run = await client.claim()
    if not run:
        return False
    logger.info(
        "claimed run", extra={"run_id": run.id, "objective": run.objective[:120]}
    )
    try:
        context = await client.get_context(run.id)
        llm_context = _context_for_llm(run, context)

        proposal_bundle = await llm.propose_hypothesis(llm_context)
        proposal_payload = await client.submit_proposal(
            run.id, proposal_bundle["proposal"]
        )

        timeout_seconds = run.budget_seconds or 3600
        result = await client.wait_for_experiment_result(
            run_id=run.id,
            timeout_seconds=timeout_seconds,
            context=context,
        )
        if result is None:
            await client.complete(
                run.id, "FAILED", reason="experiment timeout without results"
            )
            return True

        result_payload = result.model_dump(mode="json")
        decision_bundle = await llm.decide(
            context={**llm_context, "proposal_response": proposal_payload},
            proposal=proposal_bundle["proposal"],
            result=result_payload,
        )
        await client.submit_decision(run.id, decision_bundle["decision"])
        action = str(decision_bundle["decision"].get("action") or "").upper()
        if action in {"CONTINUE_RESEARCH", "MUTATE_HYPOTHESIS"} and (
            run.experiments_completed + 1 < run.budget_experiments
        ):
            # Return the lease so the same (or another) agent can pick the
            # next iteration; progress is persisted server-side.
            await client.complete(run.id, "REQUEUE", reason=action)
        else:
            await client.complete(run.id, "COMPLETED", reason=action or "DONE")
        return True
    except LeaseLostError:
        logger.warning("lease lost", extra={"run_id": run.id})
        client.drop_lease()
        return False
    except AgentAPIError as exc:
        logger.error("agent api error", extra={"run_id": run.id, "error": str(exc)})
        if exc.status_code >= 500:
            await client.complete(
                run.id, "REQUEUE", reason=f"api error {exc.status_code}"
            )
        else:
            await client.complete(run.id, "FAILED", reason=str(exc)[:400])
        return True


async def run_loop() -> None:
    base_url = __import__("os").getenv("AI_LAB_API_BASE_URL", "http://api:8001")
    token = __import__("os").getenv("AI_LAB_AGENT_TOKEN", "")
    if not token:
        raise RuntimeError("AI_LAB_AGENT_TOKEN is required")
    client = AILabApiClient(base_url, token, poll_seconds=POLL_SECONDS)
    llm = OpenCodeClient()
    stopping = asyncio.Event()

    def _stop(*_args) -> None:
        stopping.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    logger.info("ai_research_agent started")
    while not stopping.is_set():
        try:
            progressed = await process_one_run(client, llm)
            if not progressed:
                await asyncio.sleep(IDLE_SLEEP_SECONDS)
        except Exception as exc:  # noqa: BLE001 - worker must survive errors
            logger.error("iteration failed", extra={"error": str(exc)})
            await asyncio.sleep(IDLE_SLEEP_SECONDS)
    logger.info("ai_research_agent stopped")


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    await run_loop()


if __name__ == "__main__":
    asyncio.run(main())
