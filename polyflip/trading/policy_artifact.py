"""Immutable weighted-policy artifacts and activation evidence gates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Optional

from polyflip.trading.weighted_benchmark import StackerModel
from polyflip.trading.weighted_policy import WeightedPolicyConfig


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def artifact_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PolicyArtifact:
    artifact_id: str
    version: str
    created_at: str
    training_window: Mapping[str, Any]
    model: Mapping[str, Any]
    policy_config: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    source_report_hash: Optional[str] = None

    def payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "training_window": dict(self.training_window),
            "model": dict(self.model),
            "policy_config": dict(self.policy_config),
            "thresholds": dict(self.thresholds),
            "source_report_hash": self.source_report_hash,
        }

    def as_dict(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, **self.payload()}


def create_policy_artifact(
    *,
    version: str,
    created_at: str,
    training_window: Mapping[str, Any],
    stacker: Optional[StackerModel],
    policy_config: WeightedPolicyConfig,
    thresholds: Mapping[str, Any],
    source_report_hash: Optional[str] = None,
) -> PolicyArtifact:
    model = stacker.as_dict() if stacker else {"type": "NONE"}
    config = {
        key: value
        for key, value in vars(policy_config).items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    payload = {
        "version": version,
        "created_at": created_at,
        "training_window": dict(training_window),
        "model": model,
        "policy_config": config,
        "thresholds": dict(thresholds),
        "source_report_hash": source_report_hash,
    }
    return PolicyArtifact(
        artifact_id=artifact_hash(payload),
        version=version,
        created_at=created_at,
        training_window=dict(training_window),
        model=model,
        policy_config=config,
        thresholds=dict(thresholds),
        source_report_hash=source_report_hash,
    )


def save_policy_artifact(path: str | Path, artifact: PolicyArtifact) -> None:
    """Write once; refuse to overwrite a different artifact."""
    destination = Path(path)
    if destination.exists():
        existing = load_policy_artifact(destination)
        if existing.artifact_id != artifact.artifact_id:
            raise ValueError(f"immutable policy artifact already exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact.as_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def weighted_policy_config_from_artifact(
    artifact: PolicyArtifact,
    *,
    fallback: Optional[WeightedPolicyConfig] = None,
) -> WeightedPolicyConfig:
    """Convert an immutable artifact config into runtime policy settings."""
    base = fallback or WeightedPolicyConfig()
    allowed = {item.name for item in fields(WeightedPolicyConfig)}
    values = {item.name: getattr(base, item.name) for item in fields(WeightedPolicyConfig)}
    for key, value in artifact.policy_config.items():
        if key not in allowed:
            continue
        if value is not None and not isinstance(value, (str, int, float, bool)):
            continue
        values[key] = value
    values["policy_id"] = artifact.artifact_id[:64]
    return replace(base, **values)


def load_policy_artifact(path: str | Path) -> PolicyArtifact:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    artifact_id = str(raw.pop("artifact_id", ""))
    expected = artifact_hash(raw)
    if artifact_id != expected:
        raise ValueError(f"policy artifact hash mismatch: {path}")
    return PolicyArtifact(
        artifact_id=artifact_id,
        version=str(raw["version"]),
        created_at=str(raw["created_at"]),
        training_window=raw["training_window"],
        model=raw["model"],
        policy_config=raw["policy_config"],
        thresholds=raw["thresholds"],
        source_report_hash=raw.get("source_report_hash"),
    )


@dataclass(frozen=True)
class ActivationEvidence:
    shadow_days: float = 0.0
    shadow_resolved_markets: int = 0
    shadow_candidate_trades: int = 0
    repeat_oot_reports: int = 0
    live_fills: int = 0
    pnl_ci_lower: Optional[float] = None


@dataclass(frozen=True)
class ActivationGate:
    eligible: bool
    reasons: tuple[str, ...]


def activation_gate(
    evidence: ActivationEvidence,
    *,
    min_shadow_days: float = 14.0,
    min_resolved_markets: int = 1000,
    min_candidate_trades: int = 300,
    min_repeat_oot_reports: int = 1,
    min_live_fills: int = 300,
) -> ActivationGate:
    """Require plan evidence before permitting fixed-bet ACTIVE rollout."""
    reasons: list[str] = []
    if evidence.shadow_days < min_shadow_days:
        reasons.append("SHADOW_DAYS_BELOW_MINIMUM")
    if evidence.shadow_resolved_markets < min_resolved_markets:
        reasons.append("SHADOW_RESOLVED_MARKETS_BELOW_MINIMUM")
    if evidence.shadow_candidate_trades < min_candidate_trades:
        reasons.append("SHADOW_CANDIDATE_TRADES_BELOW_MINIMUM")
    if evidence.repeat_oot_reports < min_repeat_oot_reports:
        reasons.append("REPEAT_OOT_REPORTS_BELOW_MINIMUM")
    if evidence.live_fills < min_live_fills:
        reasons.append("LIVE_FILLS_BELOW_MINIMUM")
    if evidence.pnl_ci_lower is not None and evidence.pnl_ci_lower <= 0.0:
        reasons.append("PNL_CI_LOWER_NOT_POSITIVE")
    return ActivationGate(eligible=not reasons, reasons=tuple(reasons))
