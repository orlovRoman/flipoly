"""Autonomous research loop for the independent AI research agent.

One iteration: claim -> context -> hypothesis -> proposal -> wait TRAIN/OOT ->
decision -> complete. All side effects go through the AI Lab HTTP API.
Resumable: branches on server phase NEEDS_PROPOSAL/WAITING_RESULT/NEEDS_DECISION/NEEDS_COMPLETION,
heartbeat over whole run, handles lease loss, transient errors, budget exhaustion.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from api_client import AgentAPIError, AILabApiClient, LeaseLostError
from opencode_client import OpenCodeClient
from schemas import ClaimedRun

logger = logging.getLogger("ai_research_agent")

POLL_SECONDS = float(__import__("os").getenv("AI_LAB_POLL_SECONDS", "10"))
IDLE_SLEEP_SECONDS = max(POLL_SECONDS, 2.0)


async def _requeue_after_worker_error(
    client: AILabApiClient, run: ClaimedRun, exc: Exception
) -> None:
    """Finish errors outside the API contract without an unbounded retry loop.

    LLM transports can raise httpx and JSON/parsing exceptions directly.
    If those escape the phase loop, the heartbeat is stopped in finally and
    the run would otherwise remain RUNNING until its lease expires. Only
    explicit transient API errors may be requeued; worker errors are terminal
    for this attempt so an unavailable upstream cannot claim the same run forever.
    """
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    transient_api_error = (
        status_code is not None
        and (status_code >= 500 or status_code in {408, 409, 425, 429})
    )
    action = "REQUEUE" if transient_api_error else "FAILED"
    reason = f"worker error ({type(exc).__name__}): {str(exc)[:320]}"
    logger.error(
        f"unexpected worker error; {action.lower()} run",
        extra={"run_id": run.id, "error": str(exc), "exception": type(exc).__name__},
    )
    try:
        await client.complete(run.id, action, reason=reason)
    except LeaseLostError:
        logger.warning(
            "lease lost while recovering worker error", extra={"run_id": run.id}
        )
    except Exception as recovery_exc:  # noqa: BLE001 - recovery is best effort
        logger.error(
            "failed to requeue run after worker error",
            extra={"run_id": run.id, "error": str(recovery_exc)},
        )
    finally:
        client.drop_lease()


def _context_for_llm(run: ClaimedRun, context) -> dict:
    # New snapshot shape: run.llm_snapshot contains per-model protocol.
    snap = getattr(run, "llm_snapshot", None) or {}
    research = getattr(run, "llm_research", None)
    summary = getattr(run, "llm_summary", None)
    if not isinstance(research, dict):
        research = (
            snap.get("research") if isinstance(snap.get("research"), dict) else None
        )
    if not isinstance(summary, dict):
        summary = snap.get("summary") if isinstance(snap.get("summary"), dict) else None
    # Fallback to flat fields with protocol from snapshot.
    if not isinstance(research, dict):
        research = {
            "model_id": run.llm_research_model or "",
            "protocol": (snap.get("protocol") if isinstance(snap, dict) else "")
            or "responses",
        }
    if not isinstance(summary, dict):
        summary = {
            "model_id": run.llm_summary_model or "",
            "protocol": (snap.get("protocol") if isinstance(snap, dict) else "")
            or "responses",
        }
    return {
        "run_id": run.id,
        "objective": run.objective,
        "scope": run.scope,
        "autonomy_level": run.autonomy_level,
        "budget_remaining_steps": max(
            run.budget_experiments - run.experiments_completed, 0
        ),
        "research_model": research.get("model_id") or run.llm_research_model or "",
        "research": research,
        "summary_model": summary.get("model_id") or run.llm_summary_model or "",
        "summary": summary,
        "llm_snapshot": snap,
        "active_models": context.active_models,
        "recent_trade_statistics": context.recent_trade_statistics,
        "prior_experiments": context.prior_experiments,
        "available_feature_sets": context.available_feature_sets,
        "quality_gate": context.quality_gate,
    }


async def _heartbeat_loop(
    client: AILabApiClient, run_id: int, stop_event: asyncio.Event
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.sleep(POLL_SECONDS)
            if stop_event.is_set():
                break
            await client.heartbeat(run_id)
        except LeaseLostError:
            logger.warning("heartbeat lease lost", extra={"run_id": run_id})
            stop_event.set()
            break
        except asyncio.CancelledError:
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "heartbeat transient error", extra={"run_id": run_id, "error": str(exc)}
            )
            # continue looping


async def _single_iteration_fallback(
    client: AILabApiClient, llm: OpenCodeClient, run: ClaimedRun
) -> bool:
    """Legacy single-iteration path for clients without phase support (tests)."""
    try:
        if run.experiments_completed >= run.budget_experiments:
            await client.complete(
                run.id,
                "COMPLETED",
                reason="budget exhausted",
            )
            return True
        context = await client.get_context(run.id)
        llm_context = _context_for_llm(run, context)

        proposal_bundle = await llm.propose_hypothesis(llm_context)
        proposal_payload = await client.submit_proposal(
            run.id,
            proposal_bundle["proposal"],
            telemetry=proposal_bundle.get("telemetry"),
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
        await client.submit_decision(
            run.id,
            decision_bundle["decision"],
            telemetry=decision_bundle.get("telemetry"),
        )
        action = str(decision_bundle["decision"].get("action") or "").upper()
        if action in {"CONTINUE_RESEARCH", "MUTATE_HYPOTHESIS"} and (
            run.experiments_completed + 1 < run.budget_experiments
        ):
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
        # Transient 5xx -> REQUEUE
        try:
            if exc.status_code >= 500:
                await client.complete(
                    run.id, "REQUEUE", reason=f"api error {exc.status_code}"
                )
            else:
                await client.complete(run.id, "FAILED", reason=str(exc)[:400])
        except LeaseLostError:
            client.drop_lease()
            return False
        except Exception:
            pass
        return True
    except Exception as exc:  # noqa: BLE001 - an LLM/runtime error must not strand a run
        await _requeue_after_worker_error(client, run, exc)
        return False


async def process_one_run(client: AILabApiClient, llm: OpenCodeClient) -> bool:
    run = await client.claim()
    if not run:
        return False
    logger.info(
        "claimed run", extra={"run_id": run.id, "objective": run.objective[:120]}
    )
    # If client doesn't support phase (FakeClient in tests), fallback to single iteration
    if not hasattr(client, "get_phase"):
        return await _single_iteration_fallback(client, llm, run)

    stop_event = asyncio.Event()
    hb_task = asyncio.create_task(_heartbeat_loop(client, run.id, stop_event))
    try:
        # Resumable loop: branch on server phase
        for _ in range(20):  # safety bound
            try:
                phase_data = await client.get_phase(run.id)
                phase = (
                    str(phase_data.get("phase") or "NEEDS_PROPOSAL")
                    if isinstance(phase_data, dict)
                    else "NEEDS_PROPOSAL"
                )
            except LeaseLostError:
                logger.warning("lease lost on get_phase", extra={"run_id": run.id})
                client.drop_lease()
                return False
            except AgentAPIError as exc:
                if exc.status_code >= 500:
                    logger.warning(
                        "transient get_phase error",
                        extra={"run_id": run.id, "error": str(exc)},
                    )
                    await asyncio.sleep(POLL_SECONDS)
                    continue
                else:
                    logger.error(
                        "get_phase failed", extra={"run_id": run.id, "error": str(exc)}
                    )
                    try:
                        await client.complete(run.id, "FAILED", reason=str(exc)[:400])
                    except Exception:
                        pass
                    return True

            logger.info("run phase", extra={"run_id": run.id, "phase": phase})

            if phase == "NEEDS_PROPOSAL":
                if run.experiments_completed >= run.budget_experiments:
                    await client.complete(
                        run.id,
                        "COMPLETED",
                        reason="budget exhausted",
                    )
                    break
                try:
                    context = await client.get_context(run.id)
                    llm_context = _context_for_llm(run, context)
                    if isinstance(phase_data, dict):
                        llm_context["previous_decision"] = phase_data.get(
                            "latest_decision"
                        )
                    proposal_bundle = await llm.propose_hypothesis(llm_context)
                    await client.submit_proposal(
                        run.id,
                        proposal_bundle["proposal"],
                        client_request_id=(
                            f"proposal-{run.id}-{run.experiments_completed}"
                        ),
                        telemetry=proposal_bundle.get("telemetry"),
                    )
                    continue
                except LeaseLostError:
                    client.drop_lease()
                    return False
                except AgentAPIError as exc:
                    if exc.status_code >= 500:
                        await asyncio.sleep(POLL_SECONDS)
                        continue
                    try:
                        await client.complete(run.id, "FAILED", reason=str(exc)[:400])
                    except Exception:
                        pass
                    break

            elif phase == "WAITING_RESULT":
                try:
                    timeout_seconds = run.budget_seconds or 3600
                    # Re-fetch context for wait (not strictly needed)
                    try:
                        context = await client.get_context(run.id)
                    except Exception:
                        context = None
                    result = await client.wait_for_experiment_result(
                        run_id=run.id,
                        timeout_seconds=timeout_seconds,
                        context=context,
                    )
                    if result is None:
                        await client.complete(
                            run.id,
                            "FAILED",
                            reason="experiment timeout without results",
                        )
                        break
                    # Result ready, next phase will be NEEDS_DECISION
                    continue
                except LeaseLostError:
                    client.drop_lease()
                    return False
                except AgentAPIError as exc:
                    if exc.status_code >= 500:
                        await asyncio.sleep(POLL_SECONDS)
                        continue
                    try:
                        await client.complete(run.id, "FAILED", reason=str(exc)[:400])
                    except Exception:
                        pass
                    break

            elif phase == "NEEDS_DECISION":
                try:
                    context = await client.get_context(run.id)
                    llm_context = _context_for_llm(run, context)
                    # Need latest result
                    result_data = await client.get_result(run.id)
                    result_payload = None
                    if isinstance(result_data, dict) and result_data.get("result"):
                        # result_data is {"state":..., "result": {...}}
                        result_payload = result_data["result"]
                    proposal = {}
                    if isinstance(phase_data, dict):
                        latest_proposal = phase_data.get("latest_proposal")
                        if isinstance(latest_proposal, dict):
                            proposal = dict(latest_proposal)
                        if not proposal and phase_data.get("latest_config_id"):
                            proposal = {"config_id": phase_data["latest_config_id"]}
                    decision_bundle = await llm.decide(
                        context=llm_context,
                        proposal=proposal,
                        result=result_payload,
                    )
                    result_id = (
                        phase_data.get("latest_result_id")
                        if isinstance(phase_data, dict)
                        else None
                    )
                    await client.submit_decision(
                        run.id,
                        decision_bundle["decision"],
                        client_request_id=f"decision-{run.id}-{result_id}",
                        telemetry=decision_bundle.get("telemetry"),
                    )
                    # Update local run for budget
                    run.experiments_completed = (run.experiments_completed or 0) + 1
                    continue
                except LeaseLostError:
                    client.drop_lease()
                    return False
                except AgentAPIError as exc:
                    if exc.status_code >= 500:
                        await asyncio.sleep(POLL_SECONDS)
                        continue
                    try:
                        await client.complete(run.id, "FAILED", reason=str(exc)[:400])
                    except Exception:
                        pass
                    break

            elif phase == "NEEDS_COMPLETION":
                try:
                    last_action = "COMPLETED"
                    if isinstance(phase_data, dict) and isinstance(
                        phase_data.get("latest_decision"), dict
                    ):
                        last_action = str(
                            phase_data["latest_decision"].get("action") or "COMPLETED"
                        ).upper()
                    elif isinstance(phase_data, dict) and isinstance(
                        phase_data.get("latest_decision"), bool
                    ):
                        # Legacy boolean case, fallback
                        last_action = "COMPLETED"
                    remaining = max(
                        run.budget_experiments - run.experiments_completed, 0
                    )
                    if (
                        last_action in {"CONTINUE_RESEARCH", "MUTATE_HYPOTHESIS"}
                        and remaining > 0
                    ):
                        await client.complete(run.id, "REQUEUE", reason=last_action)
                    elif last_action in {"CONTINUE_RESEARCH", "MUTATE_HYPOTHESIS"}:
                        await client.complete(
                            run.id, "COMPLETED", reason="budget exhausted"
                        )
                    else:
                        await client.complete(run.id, "COMPLETED", reason=last_action)
                    break
                except LeaseLostError:
                    client.drop_lease()
                    return False
                except AgentAPIError as exc:
                    if exc.status_code >= 500:
                        await asyncio.sleep(POLL_SECONDS)
                        continue
                    try:
                        await client.complete(run.id, "FAILED", reason=str(exc)[:400])
                    except Exception:
                        pass
                    break

            else:
                # Unknown phase, fallback to single iteration
                return await _single_iteration_fallback(client, llm, run)

        else:
            try:
                await client.complete(
                    run.id,
                    "REQUEUE",
                    reason="phase loop retry limit reached",
                )
            except LeaseLostError:
                client.drop_lease()
                return False
            except AgentAPIError:
                logger.exception(
                    "failed to release run after phase loop",
                    extra={"run_id": run.id},
                )
        return True
    except LeaseLostError:
        logger.warning("lease lost", extra={"run_id": run.id})
        client.drop_lease()
        return False
    except AgentAPIError as exc:
        logger.error("agent api error", extra={"run_id": run.id, "error": str(exc)})
        try:
            if exc.status_code >= 500:
                await client.complete(
                    run.id, "REQUEUE", reason=f"api error {exc.status_code}"
                )
            else:
                await client.complete(run.id, "FAILED", reason=str(exc)[:400])
        except LeaseLostError:
            client.drop_lease()
            return False
        except Exception:
            pass
        return True
    except Exception as exc:  # noqa: BLE001 - requeue unexpected LLM/runtime failures
        await _requeue_after_worker_error(client, run, exc)
        return False
    finally:
        stop_event.set()
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass


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
