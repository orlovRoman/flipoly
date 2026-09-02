"""Cost-aware offline evaluation for the weighted trading policy.

The module is pure apart from numpy and accepts exported database rows.  It
shares the live policy's probability, side-selection, fee and slippage code so
the benchmark cannot silently use a different EV definition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from math import log
from typing import Any, Mapping, Optional, Sequence

import hashlib
import json

import numpy as np

from polyflip.trading.weighted_sizing import conservative_size, stepped_bet_size
from polyflip.trading.weighted_policy import (
    BUY_NO,
    BUY_YES,
    WeightedPolicyConfig,
    clamp_probability,
    logit,
    score_weighted_probability,
    probability_for_side,
    estimate_trade_cost,
    select_weighted_side,
)


FIXED_HORIZONS: tuple[str, ...] = ("10M", "5M", "2M")


def _dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value or "")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            result = datetime.fromisoformat(text)
        except (TypeError, ValueError):
            result = datetime.min.replace(tzinfo=timezone.utc)
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _probability(value: Any) -> Optional[float]:
    return clamp_probability(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _first_present(raw: Mapping[str, Any], *names: str) -> Any:
    """Return the first non-null/non-empty value from a compatibility set."""
    for name in names:
        value = raw.get(name)
        if value is not None and value != "":
            return value
    return None


def _evidence(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(result):
        return None
    return max(-1.0, min(1.0, result))


def normalize_horizon(value: Any) -> str:
    """Normalize exported horizon labels to the benchmark's fixed buckets."""
    text = str(value or "").strip().upper().replace("_", "").replace("-", "")
    aliases = {
        "2": "2M",
        "2M": "2M",
        "2MIN": "2M",
        "5": "5M",
        "5M": "5M",
        "5MIN": "5M",
        "10": "10M",
        "10M": "10M",
        "10MIN": "10M",
    }
    return aliases.get(text, text)

def _outcome_yes(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) in (0.0, 1.0):
            return bool(value)
    text = str(value).strip().upper()
    if text in {"YES", "UP", "TRUE", "1", "WIN", "WON"}:
        return True
    if text in {"NO", "DOWN", "FALSE", "0", "LOSS", "LOST"}:
        return False
    return None


def _action(value: Any) -> Optional[str]:
    """Normalize a persisted legacy/candidate action to the policy axis."""
    text = str(value or "").strip().upper()
    if text in {"BUY_YES", "YES", "UP"}:
        return BUY_YES
    if text in {"BUY_NO", "NO", "DOWN"}:
        return BUY_NO
    if text in {"SKIP", "NONE", "CANCELLED", "FAILED"}:
        return "SKIP"
    return None


@dataclass(frozen=True)
class MarketObservation:
    """Exactly one market/horizon observation for benchmark purposes."""

    market_id: str
    timestamp: datetime
    asset: str
    yes_ask: Optional[float]
    no_ask: Optional[float]
    outcome_yes: Optional[bool]
    p_market_yes: Optional[float] = None
    p_logreg_yes: Optional[float] = None
    p_lgbm_yes: Optional[float] = None
    p_legacy_yes: Optional[float] = None
    mrf_evidence: Optional[float] = None
    spread: float = 0.0
    fee_rate: Optional[float] = None
    fee_exponent: Optional[float] = None
    fee_source: str = "CONFIG_DEFAULT"
    execution_role: str = "TAKER"
    market_role: Optional[str] = None
    strategy_type: Optional[str] = None
    phase: Optional[str] = None
    asset_phase: Optional[str] = None
    observed_cost_per_share: Optional[float] = None
    group: Optional[str] = None
    horizon: str = ""
    time_left_sec: Optional[float] = None
    legacy_action: Optional[str] = None
    legacy_ask: Optional[float] = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MarketObservation":
        yes_ask = _first_present(raw, "yes_ask", "candidate_ask")
        no_ask = _first_present(raw, "no_ask")
        p_market = _first_present(raw, "p_market_yes", "weighted_p_market_yes")
        if p_market is None and yes_ask is not None and no_ask is not None:
            try:
                yes, no = float(yes_ask), float(no_ask)
                if yes > 0.0 and no > 0.0:
                    p_market = yes / (yes + no)
            except (TypeError, ValueError):
                pass
        raw_evidence = _first_present(raw, "mrf_evidence", "weighted_mrf_evidence")
        observed_cost = _first_present(
            raw, "observed_cost_per_share", "observed_fee_per_share"
        )
        legacy_ask = _first_present(raw, "legacy_ask", "candidate_ask")
        legacy_action = _action(
            _first_present(
                raw, "legacy_action", "final_action", "action", "candidate_side"
            )
        )
        candidate_action = _action(
            _first_present(
                raw, "candidate_side", "legacy_action", "final_action", "action"
            )
        )
        raw_p_logreg_yes = _first_present(
            raw, "p_logreg_yes", "weighted_p_logreg_yes"
        )
        p_logreg = _probability(raw_p_logreg_yes)
        if p_logreg is None:
            # Older funnel/trade rows persist p_logreg_win relative to the
            # selected candidate side, not on the YES axis.  Convert BUY_NO
            # rows before exposing the canonical p_logreg_yes field.
            candidate_win = _probability(
                _first_present(raw, "p_logreg_win")
            )
            if candidate_win is not None:
                if candidate_action == BUY_YES:
                    p_logreg = candidate_win
                elif candidate_action == BUY_NO:
                    p_logreg = 1.0 - candidate_win
        return cls(
            market_id=str(raw.get("market_id", raw.get("condition_id", ""))),
            timestamp=_dt(raw.get("timestamp", raw.get("created_at"))),
            asset=str(raw.get("asset", "")),
            yes_ask=_probability(yes_ask),
            no_ask=_probability(no_ask),
            outcome_yes=_outcome_yes(
                _first_present(raw, "outcome_yes", "final_outcome", "settlement_outcome")
            ),
            p_market_yes=_probability(p_market),
            p_logreg_yes=p_logreg,
            p_lgbm_yes=_probability(
                _first_present(raw, "p_lgbm_yes", "weighted_p_lgbm_yes")
            ),
            p_legacy_yes=_probability(raw.get("p_legacy_yes")),
            mrf_evidence=_evidence(raw_evidence),
            spread=max(0.0, _optional_float(raw.get("spread", 0.0)) or 0.0),
            fee_rate=_optional_float(raw.get("fee_rate")),
            fee_exponent=_optional_float(raw.get("fee_exponent")),
            fee_source=str(raw.get("fee_source", "CONFIG_DEFAULT") or "CONFIG_DEFAULT"),
            execution_role=str(raw.get("execution_role", "TAKER") or "TAKER").upper(),
            market_role=(
                str(raw["market_role"]).upper()
                if raw.get("market_role") is not None
                else None
            ),
            strategy_type=(
                str(raw["strategy_type"]).upper()
                if raw.get("strategy_type") is not None
                else None
            ),
            phase=(
                str(
                    raw.get(
                        "phase",
                        raw.get(
                            "market_phase",
                            raw.get("mrf_phase", raw.get("entry_model_phase")),
                        ),
                    )
                ).upper()
                if raw.get(
                    "phase",
                    raw.get(
                        "market_phase",
                        raw.get("mrf_phase", raw.get("entry_model_phase")),
                    ),
                )
                is not None
                else None
            ),
            asset_phase=(
                str(raw["asset_phase"]).upper()
                if raw.get("asset_phase") is not None
                else (
                    str(raw["mrf_asset_phase"]).upper()
                    if raw.get("mrf_asset_phase") is not None
                    else None
                )
            ),
            observed_cost_per_share=(
                max(0.0, _optional_float(observed_cost))
                if _optional_float(observed_cost) is not None
                else None
            ),
            group=str(raw.get("group", raw.get("asset", "")) or ""),
            horizon=normalize_horizon(raw.get("horizon", raw.get("market_horizon", ""))),
            time_left_sec=_optional_float(
                raw.get("time_left_sec", raw.get("time_to_close_sec"))
            ),
            legacy_action=_action(legacy_action),
            legacy_ask=_probability(legacy_ask),
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass(frozen=True)
class BenchmarkConfig:
    policy_config: WeightedPolicyConfig = field(default_factory=WeightedPolicyConfig)
    min_net_ev: float = 0.0
    train_min_rows: int = 300
    test_size: int = 100
    purge_gap: int = 0
    ridge_lambda: float = 1.0
    coefficient_bound: float = 5.0
    bootstrap_iterations: int = 1000
    bootstrap_seed: int = 20260901
    candidate_min_net_ev: tuple[float, ...] = (
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
    )
    hierarchical_min_segment_rows: int = 300
    hierarchical_shrinkage: float = 300.0
    candidate_price_caps: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40)
    candidate_favorite_price_caps: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90)
    candidate_time_windows: tuple[tuple[float, float], ...] = (
        (30.0, 300.0),
        (60.0, 600.0),
        (120.0, 900.0),
    )
    candidate_mrf_beta: tuple[float, ...] = (0.0, 0.10, 0.20, 0.25, 0.40)
    sizing_mode: str = "FIXED"
    sizing_standard_error: float = 0.0
    sizing_kelly_fraction: float = 0.025
    sizing_base_bet_usdc: float = 1.0
    sizing_cap_usdc: float = 3.0


@dataclass(frozen=True)
class PurgedFold:
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


@dataclass(frozen=True)
class StackerModel:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    training_rows: int
    ridge_lambda: float
    coefficient_bound: float
    # Number of independent markets represented by the training rows.  A
    # market can have multiple fixed-horizon snapshots, so this is tracked
    # separately from ``training_rows`` for hierarchical minimum-size gates.
    training_markets: int = 0

    def predict_one(self, observation: MarketObservation) -> Optional[float]:
        features = _stacker_features(observation)
        if features is None:
            return None
        score = float(np.asarray(features) @ np.asarray(self.coefficients))
        return float(1.0 / (1.0 + np.exp(-max(-40.0, min(40.0, score)))))

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "coefficients": list(self.coefficients),
            "training_rows": self.training_rows,
            "training_markets": self.training_markets,
            "ridge_lambda": self.ridge_lambda,
            "coefficient_bound": self.coefficient_bound,
        }


@dataclass(frozen=True)
class TradeEvaluation:
    market_id: str
    timestamp: datetime
    asset: str
    side: str
    ask: float
    p_win: float
    outcome_yes: bool
    pnl: float
    cost_per_share: float
    group: str
    size_multiplier: float = 1.0


@dataclass
class ArmMetrics:
    arm: str
    observations: int = 0
    trades: int = 0
    wins: int = 0
    net_pnl: float = 0.0
    total_cost: float = 0.0
    brier: Optional[float] = None
    log_loss: Optional[float] = None
    pnl_ci_low: Optional[float] = None
    pnl_ci_high: Optional[float] = None
    evaluations: list[TradeEvaluation] = field(default_factory=list)

    @property
    def win_rate(self) -> Optional[float]:
        return self.wins / self.trades if self.trades else None

    @property
    def mean_pnl(self) -> Optional[float]:
        return self.net_pnl / self.trades if self.trades else None

    @property
    def exposure(self) -> float:
        return sum(
            max(0.0, float(item.ask)) * max(0.0, float(item.size_multiplier))
            for item in self.evaluations
        )

    @property
    def roi(self) -> Optional[float]:
        denominator = self.exposure
        return self.net_pnl / denominator if denominator > 0.0 else None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["win_rate"] = self.win_rate
        result["mean_pnl"] = self.mean_pnl
        result["exposure"] = round(self.exposure, 10)
        result["roi"] = self.roi
        result.pop("evaluations", None)
        return result


@dataclass(frozen=True)
class BenchmarkReport:
    generated_at: str
    observations: int
    resolved_observations: int
    folds: tuple[PurgedFold, ...]
    stacker: Optional[StackerModel]
    arms: tuple[ArmMetrics, ...]
    sensitivity: tuple[dict[str, Any], ...]
    hierarchical_stacker: Optional["HierarchicalStacker"] = None
    duplicate_rows_removed: int = 0
    dataset_fingerprint: str = ""
    oof_standard_error: Optional[float] = None
    stability: tuple[dict[str, Any], ...] = ()
    sizing_steps: tuple[dict[str, Any], ...] = ()
    kelly_fractions: tuple[dict[str, Any], ...] = ()
    tuning: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "observations": self.observations,
            "resolved_observations": self.resolved_observations,
            "folds": [
                {
                    "train_indices": list(fold.train_indices),
                    "test_indices": list(fold.test_indices),
                }
                for fold in self.folds
            ],
            "stacker": self.stacker.as_dict() if self.stacker else None,
            "arms": [arm.as_dict() for arm in self.arms],
            "sensitivity": list(self.sensitivity),
            "hierarchical_stacker": self.hierarchical_stacker.as_dict() if self.hierarchical_stacker else None,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "oof_standard_error": self.oof_standard_error,
            "stability": [dict(item) for item in self.stability],
            "sizing_steps": [dict(item) for item in self.sizing_steps],
            "kelly_fractions": [dict(item) for item in self.kelly_fractions],
            "tuning": [dict(item) for item in self.tuning],
        }


def purged_walk_forward_folds(
    observations: Sequence[MarketObservation],
    *,
    train_min_rows: int = 300,
    test_size: int = 100,
    purge_gap: int = 0,
) -> tuple[PurgedFold, ...]:
    """Build chronological folds without splitting one market across windows.

    purge_gap is expressed in market groups, not raw rows.  A market can have
    several fixed-horizon observations, so purging rows alone would leak
    another horizon of the same market into training.
    """
    order = sorted(
        range(len(observations)),
        key=lambda i: (observations[i].timestamp, observations[i].market_id, i),
    )
    train_min_rows, test_size = max(1, int(train_min_rows)), max(1, int(test_size))
    purge_gap = max(0, int(purge_gap))
    grouped: dict[str, list[int]] = {}
    group_order: list[str] = []
    for index in order:
        key = observations[index].market_id or f"__row__{index}"
        if key not in grouped:
            grouped[key] = []
            group_order.append(key)
        grouped[key].append(index)
    if not group_order:
        return ()

    train_groups = 0
    train_rows = 0
    while train_groups < len(group_order) and train_rows < train_min_rows:
        train_rows += len(grouped[group_order[train_groups]])
        train_groups += 1

    folds: list[PurgedFold] = []
    test_start = train_groups + purge_gap
    while test_start < len(group_order):
        train_end = max(0, test_start - purge_gap)
        test_end = test_start
        test_rows = 0
        while test_end < len(group_order) and (
            test_rows < test_size or test_end == test_start
        ):
            test_rows += len(grouped[group_order[test_end]])
            test_end += 1
        folds.append(
            PurgedFold(
                train_indices=tuple(
                    index
                    for key in group_order[:train_end]
                    for index in grouped[key]
                ),
                test_indices=tuple(
                    index
                    for key in group_order[test_start:test_end]
                    for index in grouped[key]
                ),
            )
        )
        test_start = test_end
    return tuple(folds)

def _is_outsider_role(role: Optional[str]) -> bool:
    return str(role or "").strip().upper() in {"OUTSIDER", "OUTS", "UNDERDOG"}


def _models_agree(observation: MarketObservation) -> Optional[bool]:
    if observation.p_logreg_yes is None or observation.p_lgbm_yes is None:
        return None
    return (observation.p_logreg_yes >= 0.5) == (observation.p_lgbm_yes >= 0.5)


def _independent_market_key(observation: MarketObservation) -> str:
    """Return the identity used for independent-market sample-size gates."""
    return observation.market_id or f"{observation.asset}|{observation.timestamp.isoformat()}"


def observation_segment_key(observation: MarketObservation) -> str:
    raw_role = str(observation.market_role or "UNKNOWN").strip().upper() or "UNKNOWN"
    role = (
        "OUTSIDER"
        if _is_outsider_role(raw_role)
        else "FAVORITE"
        if raw_role in {"FAVORITE", "FAV"}
        else raw_role
    )
    phase = str(observation.phase or "UNKNOWN").strip().upper() or "UNKNOWN"
    agreement = _models_agree(observation)
    agreement_label = "AGREE" if agreement is True else ("DISAGREE" if agreement is False else "UNKNOWN")
    raw_asset = str(observation.asset or "UNKNOWN").strip().upper() or "UNKNOWN"
    raw_asset = raw_asset.split("_", 1)[0] or "UNKNOWN"
    asset = raw_asset[:-4] if raw_asset.endswith("USDT") and len(raw_asset) > 4 else raw_asset
    return f"{asset}|{phase}|{role}|{agreement_label}"

def _stacker_features(observation: MarketObservation) -> Optional[tuple[float, ...]]:
    if observation.p_market_yes is None:
        return None
    market_logit = logit(observation.p_market_yes)
    agreement = _models_agree(observation)
    role_outsider = 1.0 if _is_outsider_role(observation.market_role) else 0.0
    models_agree = 1.0 if agreement is True else 0.0
    logreg_residual = (
        logit(observation.p_logreg_yes) - market_logit
        if observation.p_logreg_yes is not None
        else 0.0
    )
    lgbm_residual = (
        logit(observation.p_lgbm_yes) - market_logit
        if observation.p_lgbm_yes is not None
        else 0.0
    )
    signed_evidence = _evidence(observation.mrf_evidence) or 0.0
    return (
        1.0,
        market_logit,
        logreg_residual,
        lgbm_residual,
        signed_evidence,
        role_outsider,
        models_agree,
        role_outsider * models_agree,
        role_outsider * logreg_residual,
        role_outsider * lgbm_residual,
    )


def fit_ridge_logistic_stacker(
    observations: Sequence[MarketObservation],
    *,
    ridge_lambda: float = 1.0,
    coefficient_bound: float = 5.0,
    max_iter: int = 100,
) -> StackerModel:
    """Fit a bounded ridge-logistic stacker using only supplied rows.

    ``market_logit`` is a fixed prior with coefficient one.  Only the model
    residuals, MRF and interaction terms are learned and bounded; this keeps a
    stacker from silently replacing the executable market prior with a second
    unconstrained market model.
    """
    rows = [
        (_stacker_features(item), 1.0 if item.outcome_yes else 0.0)
        for item in observations
        if item.outcome_yes is not None and _stacker_features(item) is not None
    ]
    if not rows:
        raise ValueError("no resolved rows with a market probability")
    x = np.asarray([row[0] for row in rows], dtype=float)
    y = np.asarray([row[1] for row in rows], dtype=float)
    beta = np.zeros(x.shape[1], dtype=float)
    market_prior_index = 1
    beta[market_prior_index] = 1.0
    penalty = max(0.0, float(ridge_lambda))
    bound = max(0.1, float(coefficient_bound))
    free_indices = np.asarray(
        [index for index in range(x.shape[1]) if index != market_prior_index],
        dtype=int,
    )
    for _ in range(max(1, int(max_iter))):
        logits = np.clip(x @ beta, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
        gradient = x.T @ (probabilities - y) + penalty * beta
        hessian = (x.T * weights) @ x
        ridge = np.eye(x.shape[1], dtype=float) * penalty
        ridge[0, 0] = 0.0
        hessian += ridge + np.eye(x.shape[1]) * 1e-8
        free_hessian = hessian[np.ix_(free_indices, free_indices)]
        free_gradient = gradient[free_indices]
        try:
            step = np.linalg.solve(free_hessian, free_gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(free_hessian) @ free_gradient
        updated = beta.copy()
        updated[free_indices] = np.clip(
            beta[free_indices] - step,
            -bound,
            bound,
        )
        updated[market_prior_index] = 1.0
        if np.max(np.abs(updated - beta)) < 1e-7:
            beta = updated
            break
        beta = updated
    training_markets = len(
        {
            _independent_market_key(item)
            for item in observations
            if item.outcome_yes is not None and _stacker_features(item) is not None
        }
    )
    return StackerModel(
        feature_names=(
            "intercept",
            "market_logit",
            "logreg_residual",
            "lgbm_residual",
            "mrf_evidence",
            "role_outsider",
            "models_agree",
            "outsider_agree",
            "outsider_logreg_residual",
            "outsider_lgbm_residual",
        ),
        coefficients=tuple(round(float(value), 10) for value in beta),
        training_rows=len(rows),
        ridge_lambda=penalty,
        coefficient_bound=bound,
        training_markets=training_markets,
    )


def _inputs(
    observation: MarketObservation,
    arm: str,
    stacker: Optional[StackerModel],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    arm = arm.upper()
    if arm == "STACKER":
        p = stacker.predict_one(observation) if stacker else None
        return p, None, None, p
    if arm == "MARKET_ONLY":
        return observation.p_market_yes, None, None, None
    if arm == "LEGACY":
        p = observation.p_legacy_yes
        if p is None:
            p = observation.p_logreg_yes
        return p, None, None, None
    if arm == "MARKET_LOGREG":
        return observation.p_market_yes, observation.p_logreg_yes, None, None
    if arm == "MARKET_LGBM":
        return observation.p_market_yes, None, observation.p_lgbm_yes, None
    if arm in {"FULL_WEIGHTED", "FULL_WEIGHTED_MRF", "WEIGHTED"}:
        return (
            observation.p_market_yes,
            observation.p_logreg_yes,
            observation.p_lgbm_yes,
            None,
        )
    if arm in {"OUTSIDER_AGREE", "OUTSIDER_AGREE_ONLY"}:
        if _observation_role(observation) != "OUTSIDER":
            return None, None, None, None
        if observation.p_logreg_yes is None or observation.p_lgbm_yes is None:
            return None, None, None, None
        if (observation.p_logreg_yes >= 0.5) != (observation.p_lgbm_yes >= 0.5):
            return None, None, None, None
        return (
            observation.p_market_yes,
            observation.p_logreg_yes,
            observation.p_lgbm_yes,
            None,
        )
    raise ValueError(f"unknown benchmark arm: {arm}")


def _arm_config(base: WeightedPolicyConfig, arm: str) -> WeightedPolicyConfig:
    common = {
        "fee_rate": base.fee_rate,
        "maker_fee_rate": base.maker_fee_rate,
        "fee_exponent": base.fee_exponent,
        "slippage_rate": base.slippage_rate,
        "latency_buffer": base.latency_buffer,
        "execution_role": base.execution_role,
        "mrf_beta": base.mrf_beta,
        "intercept": base.intercept,
        "mrf_extreme_veto_threshold": base.mrf_extreme_veto_threshold,
        "policy_id": base.policy_id,
    }
    if arm.upper() == "MARKET_ONLY":
        return WeightedPolicyConfig(market_weight=1.0, logreg_weight=0.0, lgbm_weight=0.0, **common)
    if arm.upper() == "MARKET_LOGREG":
        return WeightedPolicyConfig(market_weight=0.8, logreg_weight=0.2, lgbm_weight=0.0, **common)
    if arm.upper() == "MARKET_LGBM":
        return WeightedPolicyConfig(market_weight=0.8, logreg_weight=0.0, lgbm_weight=0.2, **common)
    return base


def _observation_role(observation: MarketObservation) -> str:
    explicit = str(observation.market_role or "").strip().upper()
    if _is_outsider_role(explicit):
        return "OUTSIDER"
    if explicit:
        return "FAVORITE" if explicit in {"FAVORITE", "FAV"} else explicit
    return "UNKNOWN"


def _quote_role(ask: float) -> str:
    return "OUTSIDER" if float(ask) < 0.50 else "FAVORITE"


@dataclass(frozen=True)
class ParameterTuneResult:
    parameter: str
    selected: Any
    stable_folds: int
    minimum_stable_folds: int
    candidates: tuple[dict[str, Any], ...]

    @property
    def stable(self) -> bool:
        return self.stable_folds >= self.minimum_stable_folds

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "selected": self.selected,
            "stable_folds": self.stable_folds,
            "minimum_stable_folds": self.minimum_stable_folds,
            "stable": self.stable,
            "candidates": [dict(item) for item in self.candidates],
        }


def evaluate_arm(
    observations: Sequence[MarketObservation],
    arm: str,
    *,
    config: WeightedPolicyConfig | None = None,
    min_net_ev: float = 0.0,
    min_net_ev_favorite: Optional[float] = None,
    min_net_ev_outsider: Optional[float] = None,
    outsider_max_price: Optional[float] = None,
    favorite_max_price: Optional[float] = None,
    time_left_range: Optional[tuple[float, float]] = None,
    time_left_role: Optional[str] = None,
    mrf_stake_gamma: Optional[float] = None,
    sizing_mode: str = "FIXED",
    sizing_standard_error: float = 0.0,
    sizing_kelly_fraction: float = 0.025,
    sizing_base_bet_usdc: float = 1.0,
    sizing_cap_usdc: float = 3.0,
    stacker: Optional[StackerModel] = None,
    stacker_predictions: Optional[Mapping[int, float]] = None,
    use_stacker_predictions: bool = False,
    evaluation_indices: Optional[Sequence[int]] = None,
) -> ArmMetrics:
    """Evaluate one arm using net realized PnL per one-share trade."""
    base = config or WeightedPolicyConfig()
    arm_name = arm.upper()
    metrics = ArmMetrics(arm=arm_name)
    brier, logloss = [], []
    allowed_indices = (
        {int(index) for index in evaluation_indices}
        if evaluation_indices is not None
        else None
    )
    for index, observation in enumerate(observations):
        if allowed_indices is not None and index not in allowed_indices:
            continue
        if observation.outcome_yes is None:
            continue
        observation_role = _observation_role(observation)
        if time_left_range is not None and (
            time_left_role is None
            or observation_role == str(time_left_role).strip().upper()
        ):
            if observation.time_left_sec is not None:
                lower, upper = time_left_range
                if (
                    float(observation.time_left_sec) < float(lower)
                    or float(observation.time_left_sec) > float(upper)
                ):
                    continue
            elif time_left_role is None:
                continue
        metrics.observations += 1
        if arm_name == "LEGACY":
            # Replay the persisted legacy action instead of routing the row
            # through the weighted selector. This keeps the benchmark's
            # control arm a true counterfactual of what actually ran.
            p_yes = clamp_probability(
                observation.p_legacy_yes
                if observation.p_legacy_yes is not None
                else observation.p_logreg_yes
            )
            if p_yes is None:
                continue
            outcome_value = 1.0 if observation.outcome_yes else 0.0
            brier.append((p_yes - outcome_value) ** 2)
            logloss.append(
                -log(max(1e-12, p_yes if observation.outcome_yes else 1.0 - p_yes))
            )
            action = observation.legacy_action
            if action not in {BUY_YES, BUY_NO}:
                continue
            ask = observation.legacy_ask
            if ask is None:
                ask = observation.yes_ask if action == BUY_YES else observation.no_ask
            ask = clamp_probability(ask)
            if ask is None or not 0.0 < ask < 1.0:
                continue
            p_win = probability_for_side(p_yes, action)
            if p_win is None:
                continue
            cost = observation.observed_cost_per_share
            if cost is None:
                cost = estimate_trade_cost(
                    ask,
                    fee_rate=base.fee_rate,
                    maker_fee_rate=base.maker_fee_rate,
                    fee_exponent=base.fee_exponent,
                    slippage_rate=base.slippage_rate,
                    latency_buffer=base.latency_buffer,
                    role=base.execution_role,
                ).total_per_share
            won = observation.outcome_yes if action == BUY_YES else not observation.outcome_yes
            try:
                base_stake = max(0.0, float(sizing_base_bet_usdc))
            except (TypeError, ValueError, OverflowError):
                base_stake = 1.0
            if base_stake <= 0.0:
                continue
            raw_pnl = (1.0 if won else 0.0) - ask - float(cost)
            pnl = raw_pnl * base_stake
            metrics.evaluations.append(
                TradeEvaluation(
                    market_id=observation.market_id,
                    timestamp=observation.timestamp,
                    asset=observation.asset,
                    side=action,
                    ask=ask,
                    p_win=p_win,
                    outcome_yes=observation.outcome_yes,
                    pnl=round(pnl, 10),
                    cost_per_share=round(float(cost) * base_stake, 10),
                    group=observation.group or observation.asset or observation.market_id,
                    size_multiplier=round(base_stake, 10),
                )
            )
            metrics.trades += 1
            metrics.wins += int(won)
            metrics.net_pnl += pnl
            metrics.total_cost += float(cost) * base_stake
            continue
        prediction_override = (
            stacker_predictions is not None
            and (use_stacker_predictions or arm_name == "STACKER")
        )
        if prediction_override:
            if use_stacker_predictions and arm_name == "OUTSIDER_AGREE_ONLY":
                if (
                    observation_role != "OUTSIDER"
                    or _models_agree(observation) is not True
                ):
                    continue
            p = stacker_predictions.get(index)
            if p is None:
                continue
            inputs = (p, None, None, p)
        else:
            inputs = _inputs(observation, arm_name, stacker)
        p_market, p_logreg, p_lgbm, p_direct = inputs
        if p_direct is not None:
            p_final = clamp_probability(p_direct)
            if p_final is None:
                continue
            policy = _arm_config(base, "MARKET_ONLY")
        else:
            if p_market is None:
                continue
            policy = _arm_config(base, arm_name)
            scored = score_weighted_probability(
                p_market_yes=p_market,
                p_logreg_yes=p_logreg,
                p_lgbm_yes=p_lgbm,
                config=policy,
                mrf_evidence=(
                    observation.mrf_evidence
                    if arm_name in {"FULL_WEIGHTED_MRF", "WEIGHTED", "OUTSIDER_AGREE", "OUTSIDER_AGREE_ONLY"}
                    else None
                ),
                asset=observation.asset,
                phase=observation.phase,
                role=observation_role,
                time_left_sec=observation.time_left_sec,
            )
            p_final = scored.p_final_yes
        outcome = 1.0 if observation.outcome_yes else 0.0
        brier.append((p_final - outcome) ** 2)
        logloss.append(-log(max(1e-12, p_final if observation.outcome_yes else 1.0 - p_final)))
        mrf = (
            observation.mrf_evidence
            if arm_name in {"FULL_WEIGHTED_MRF", "WEIGHTED", "OUTSIDER_AGREE", "OUTSIDER_AGREE_ONLY"}
            else None
        )
        selector_min = (
            0.0
            if min_net_ev_favorite is not None or min_net_ev_outsider is not None
            else min_net_ev
        )
        selected = select_weighted_side(
            p_market_yes=p_market if p_direct is None else p_final,
            p_logreg_yes=p_logreg if p_direct is None else None,
            p_lgbm_yes=p_lgbm if p_direct is None else None,
            yes_ask=observation.yes_ask,
            no_ask=observation.no_ask,
            config=policy,
            mrf_evidence=mrf,
            asset=observation.asset,
            phase=observation.phase,
            role=observation_role,
            time_left_sec=observation.time_left_sec,
            min_net_ev=selector_min,
            fee_source=observation.fee_source,
            spread=observation.spread,
            mrf_extreme_veto_threshold=policy.mrf_extreme_veto_threshold,
        )
        if selected.selected is None:
            continue
        quote = selected.selected
        quote_role = _quote_role(quote.ask)
        role_threshold = max(
            float(min_net_ev),
            float(
                min_net_ev_outsider
                if quote_role == "OUTSIDER" and min_net_ev_outsider is not None
                else min_net_ev_favorite
                if quote_role == "FAVORITE" and min_net_ev_favorite is not None
                else 0.0
            ),
        )
        if quote.net_ev_per_share < role_threshold:
            continue
        if quote_role == "OUTSIDER" and outsider_max_price is not None and quote.ask > float(outsider_max_price):
            continue
        if quote_role == "FAVORITE" and favorite_max_price is not None and quote.ask > float(favorite_max_price):
            continue
        if time_left_range is not None and (
            time_left_role is None
            or quote_role == str(time_left_role).strip().upper()
        ):
            if observation.time_left_sec is None:
                continue
            lower, upper = time_left_range
            if (
                float(observation.time_left_sec) < float(lower)
                or float(observation.time_left_sec) > float(upper)
            ):
                continue
        won = observation.outcome_yes if quote.side == BUY_YES else not observation.outcome_yes
        cost = (
            observation.observed_cost_per_share
            if observation.observed_cost_per_share is not None
            else quote.cost.total_per_share
        )
        try:
            base_stake = max(0.0, float(sizing_base_bet_usdc))
        except (TypeError, ValueError, OverflowError):
            base_stake = 1.0
        mode = str(sizing_mode or "FIXED").strip().upper()
        if mode == "LOWER_BOUND_KELLY":
            sizing = conservative_size(
                quote.p_win,
                price=quote.ask,
                cost_per_share=cost,
                standard_error=sizing_standard_error,
                fraction=sizing_kelly_fraction,
                min_edge_lower=role_threshold,
            )
            size_multiplier = sizing.size_multiplier * base_stake
        elif mode == "STEPPED_EDGE":
            sizing = conservative_size(
                quote.p_win,
                price=quote.ask,
                cost_per_share=cost,
                standard_error=sizing_standard_error,
                fraction=0.0,
                min_edge_lower=-1.0,
            )
            stepped = stepped_bet_size(
                sizing.edge_lower,
                base_bet_usdc=1.0,
                cap_usdc=sizing_cap_usdc,
            )
            size_multiplier = stepped * base_stake
        else:
            size_multiplier = base_stake
        if size_multiplier <= 0.0:
            continue
        raw_pnl = (1.0 if won else 0.0) - quote.ask - cost
        try:
            if mrf_stake_gamma is None:
                application = str(base.mrf_application or "PROBABILITY").strip().upper()
                gamma = (
                    float(base.mrf_sizing_gamma)
                    if application in {"STAKE", "STAKE_ADJUSTMENT"}
                    else 0.0
                )
            else:
                # Explicit gamma is used by compare_mrf_application() to
                # evaluate the stake-adjustment counterfactual.
                gamma = float(mrf_stake_gamma)
        except (TypeError, ValueError, OverflowError):
            gamma = 0.0
        evidence = float(observation.mrf_evidence or 0.0)
        mrf_multiplier = max(0.5, min(1.25, 1.0 + gamma * evidence))
        size_multiplier *= mrf_multiplier
        pnl = raw_pnl * size_multiplier
        metrics.evaluations.append(
            TradeEvaluation(
                market_id=observation.market_id,
                timestamp=observation.timestamp,
                asset=observation.asset,
                side=quote.side,
                ask=quote.ask,
                p_win=quote.p_win,
                outcome_yes=observation.outcome_yes,
                pnl=round(pnl, 10),
                cost_per_share=round(cost * size_multiplier, 10),
                group=observation.group or observation.asset or observation.market_id,
                size_multiplier=round(size_multiplier, 10),
            )
        )
        metrics.trades += 1
        metrics.wins += int(won)
        metrics.net_pnl += pnl
        metrics.total_cost += cost * size_multiplier
    metrics.net_pnl = round(metrics.net_pnl, 10)
    metrics.total_cost = round(metrics.total_cost, 10)
    metrics.brier = round(float(np.mean(brier)), 10) if brier else None
    metrics.log_loss = round(float(np.mean(logloss)), 10) if logloss else None
    return metrics


def _tuning_indices(
    folds: Optional[Sequence[PurgedFold]],
    evaluation_indices: Optional[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if folds:
        return tuple(tuple(fold.test_indices) for fold in folds)
    if evaluation_indices is None:
        return ((),)
    return (tuple(int(index) for index in evaluation_indices),)


def _select_tuning_candidate(
    parameter: str,
    candidates: list[dict[str, Any]],
    *,
    minimum_stable_folds: int,
) -> ParameterTuneResult:
    if not candidates:
        return ParameterTuneResult(
            parameter=parameter,
            selected=None,
            stable_folds=0,
            minimum_stable_folds=minimum_stable_folds,
            candidates=(),
        )
    selected = max(
        candidates,
        key=lambda item: (
            bool(item["stable_folds"] >= minimum_stable_folds),
            float(item["net_pnl"]),
            int(item["stable_folds"]),
            -int(item["trades"]),
        ),
    )
    return ParameterTuneResult(
        parameter=parameter,
        selected=selected["value"],
        stable_folds=int(selected["stable_folds"]),
        minimum_stable_folds=minimum_stable_folds,
        candidates=tuple(candidates),
    )


def optimize_min_net_ev(
    observations: Sequence[MarketObservation],
    *,
    role: str,
    arm: str = "FULL_WEIGHTED_MRF",
    candidate_values: Sequence[float] = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08),
    config: WeightedPolicyConfig | None = None,
    folds: Optional[Sequence[PurgedFold]] = None,
    evaluation_indices: Optional[Sequence[int]] = None,
    minimum_stable_folds: int = 3,
    stacker_predictions: Optional[Mapping[int, float]] = None,
    use_stacker_predictions: bool = False,
) -> ParameterTuneResult:
    target = str(role).strip().upper()
    if target not in {"FAVORITE", "OUTSIDER"}:
        raise ValueError("role must be FAVORITE or OUTSIDER")
    candidates: list[dict[str, Any]] = []
    for value in candidate_values:
        fold_metrics = [
            evaluate_arm(
                observations,
                arm,
                config=config,
                min_net_ev_favorite=float(value) if target == "FAVORITE" else None,
                min_net_ev_outsider=float(value) if target == "OUTSIDER" else None,
                stacker_predictions=stacker_predictions,
                use_stacker_predictions=use_stacker_predictions,
                evaluation_indices=indices or None,
            )
            for indices in _tuning_indices(folds, evaluation_indices)
        ]
        candidates.append(
            {
                "value": float(value),
                "net_pnl": round(sum(item.net_pnl for item in fold_metrics), 10),
                "trades": sum(item.trades for item in fold_metrics),
                "stable_folds": sum(1 for item in fold_metrics if item.net_pnl > 0.0),
                "folds": len(fold_metrics),
            }
        )
    return _select_tuning_candidate(
        "min_net_ev_" + target.lower(),
        candidates,
        minimum_stable_folds=minimum_stable_folds,
    )


def optimize_price_cap(
    observations: Sequence[MarketObservation],
    *,
    role: str,
    arm: str = "FULL_WEIGHTED_MRF",
    candidate_values: Sequence[float] = (0.20, 0.25, 0.30, 0.35, 0.40),
    config: WeightedPolicyConfig | None = None,
    folds: Optional[Sequence[PurgedFold]] = None,
    evaluation_indices: Optional[Sequence[int]] = None,
    minimum_stable_folds: int = 3,
    stacker_predictions: Optional[Mapping[int, float]] = None,
    use_stacker_predictions: bool = False,
) -> ParameterTuneResult:
    target = str(role).strip().upper()
    if target not in {"FAVORITE", "OUTSIDER"}:
        raise ValueError("role must be FAVORITE or OUTSIDER")
    candidates: list[dict[str, Any]] = []
    for value in candidate_values:
        fold_metrics = [
            evaluate_arm(
                observations,
                arm,
                config=config,
                outsider_max_price=float(value) if target == "OUTSIDER" else None,
                favorite_max_price=float(value) if target == "FAVORITE" else None,
                stacker_predictions=stacker_predictions,
                use_stacker_predictions=use_stacker_predictions,
                evaluation_indices=indices or None,
            )
            for indices in _tuning_indices(folds, evaluation_indices)
        ]
        candidates.append(
            {
                "value": float(value),
                "net_pnl": round(sum(item.net_pnl for item in fold_metrics), 10),
                "trades": sum(item.trades for item in fold_metrics),
                "stable_folds": sum(1 for item in fold_metrics if item.net_pnl > 0.0),
                "folds": len(fold_metrics),
            }
        )
    return _select_tuning_candidate(
        ("outsider" if target == "OUTSIDER" else "favorite") + "_max_price",
        candidates,
        minimum_stable_folds=minimum_stable_folds,
    )


def optimize_time_window(
    observations: Sequence[MarketObservation],
    *,
    role: str,
    windows: Sequence[tuple[float, float]],
    arm: str = "FULL_WEIGHTED_MRF",
    config: WeightedPolicyConfig | None = None,
    folds: Optional[Sequence[PurgedFold]] = None,
    evaluation_indices: Optional[Sequence[int]] = None,
    minimum_stable_folds: int = 3,
    stacker_predictions: Optional[Mapping[int, float]] = None,
    use_stacker_predictions: bool = False,
) -> ParameterTuneResult:
    target = str(role).strip().upper()
    if target not in {"FAVORITE", "OUTSIDER"}:
        raise ValueError("role must be FAVORITE or OUTSIDER")
    candidates: list[dict[str, Any]] = []
    for lower, upper in windows:
        fold_metrics = [
            evaluate_arm(
                observations,
                arm,
                config=config,
                time_left_range=(float(lower), float(upper)),
                time_left_role=target,
                stacker_predictions=stacker_predictions,
                use_stacker_predictions=use_stacker_predictions,
                evaluation_indices=indices or None,
            )
            for indices in _tuning_indices(folds, evaluation_indices)
        ]
        candidates.append(
            {
                "value": [float(lower), float(upper)],
                "net_pnl": round(sum(item.net_pnl for item in fold_metrics), 10),
                "trades": sum(item.trades for item in fold_metrics),
                "stable_folds": sum(1 for item in fold_metrics if item.net_pnl > 0.0),
                "folds": len(fold_metrics),
            }
        )
    return _select_tuning_candidate(
        "time_left_" + target.lower(),
        candidates,
        minimum_stable_folds=minimum_stable_folds,
    )


def optimize_mrf_beta(
    observations: Sequence[MarketObservation],
    *,
    candidate_values: Sequence[float] = (0.0, 0.10, 0.20, 0.25, 0.40),
    config: WeightedPolicyConfig | None = None,
    folds: Optional[Sequence[PurgedFold]] = None,
    evaluation_indices: Optional[Sequence[int]] = None,
    min_net_ev: float = 0.0,
    minimum_stable_folds: int = 3,
) -> ParameterTuneResult:
    """Choose probability MRF beta using OOT net PnL stability."""
    candidates: list[dict[str, Any]] = []
    base = config or WeightedPolicyConfig()
    for value in candidate_values:
        try:
            beta = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        fold_metrics = [
            evaluate_arm(
                observations,
                "FULL_WEIGHTED_MRF",
                config=replace(base, mrf_beta=beta),
                min_net_ev=min_net_ev,
                evaluation_indices=indices or None,
            )
            for indices in _tuning_indices(folds, evaluation_indices)
        ]
        candidates.append(
            {
                "value": beta,
                "net_pnl": round(sum(item.net_pnl for item in fold_metrics), 10),
                "trades": sum(item.trades for item in fold_metrics),
                "stable_folds": sum(1 for item in fold_metrics if item.net_pnl > 0.0),
                "folds": len(fold_metrics),
            }
        )
    return _select_tuning_candidate(
        "mrf_beta",
        candidates,
        minimum_stable_folds=minimum_stable_folds,
    )


def cluster_bootstrap_difference_ci(
    baseline: Sequence[TradeEvaluation],
    candidate: Sequence[TradeEvaluation],
    *,
    iterations: int = 1000,
    seed: int = 20260902,
    alpha: float = 0.05,
) -> tuple[Optional[float], Optional[float]]:
    """Return a cluster-bootstrap CI for candidate minus baseline PnL.

    Both arms are evaluated on the same OOT rows, but may trade different
    subsets. Missing arm/cluster PnL is therefore treated as zero and the
    resampling unit remains ``market_id|UTC-date`` rather than an individual
    horizon snapshot.
    """
    grouped: dict[str, list[float]] = {}
    for item, sign in ((baseline, -1.0), (candidate, 1.0)):
        for evaluation in item:
            timestamp = evaluation.timestamp.astimezone(timezone.utc)
            cluster = f"{evaluation.market_id}|{timestamp.date().isoformat()}"
            grouped.setdefault(cluster, []).append(sign * float(evaluation.pnl))
    if not grouped:
        return None, None
    keys = list(grouped)
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [
            sum(sum(grouped[keys[index]]) for index in rng.integers(0, len(keys), len(keys)))
            for _ in range(max(1, int(iterations)))
        ],
        dtype=float,
    )
    lower_q = max(0.0, min(1.0, float(alpha) / 2.0))
    upper_q = max(0.0, min(1.0, 1.0 - float(alpha) / 2.0))
    return (
        round(float(np.quantile(samples, lower_q)), 10),
        round(float(np.quantile(samples, upper_q)), 10),
    )


def compare_outsider_agreement(
    observations: Sequence[MarketObservation],
    *,
    config: WeightedPolicyConfig | None = None,
    folds: Optional[Sequence[PurgedFold]] = None,
    evaluation_indices: Optional[Sequence[int]] = None,
    min_net_ev: float = 0.0,
    stacker_predictions: Optional[Mapping[int, float]] = None,
    use_stacker_predictions: bool = False,
) -> dict[str, Any]:
    """Compare hard outsider consensus to the soft models_agree coefficient."""
    candidates: list[dict[str, Any]] = []
    metrics_by_arm: dict[str, list[ArmMetrics]] = {}
    for arm in ("FULL_WEIGHTED_MRF", "OUTSIDER_AGREE_ONLY"):
        fold_metrics = [
            evaluate_arm(
                observations,
                arm,
                config=config,
                min_net_ev=min_net_ev,
                stacker_predictions=stacker_predictions,
                use_stacker_predictions=use_stacker_predictions,
                evaluation_indices=indices or None,
            )
            for indices in _tuning_indices(folds, evaluation_indices)
        ]
        brier_values = [item.brier for item in fold_metrics if item.brier is not None]
        candidates.append(
            {
                "value": arm,
                "net_pnl": round(sum(item.net_pnl for item in fold_metrics), 10),
                "trades": sum(item.trades for item in fold_metrics),
                "stable_folds": sum(1 for item in fold_metrics if item.net_pnl > 0.0),
                "folds": len(fold_metrics),
                "brier": (
                    round(sum(brier_values) / len(brier_values), 10)
                    if brier_values
                    else None
                ),
            }
        )
        metrics_by_arm[arm] = fold_metrics
    soft_evaluations = [
        item
        for metrics in metrics_by_arm["FULL_WEIGHTED_MRF"]
        for item in metrics.evaluations
    ]
    hard_evaluations = [
        item
        for metrics in metrics_by_arm["OUTSIDER_AGREE_ONLY"]
        for item in metrics.evaluations
    ]
    delta_low, delta_high = cluster_bootstrap_difference_ci(
        soft_evaluations,
        hard_evaluations,
    )
    hard_statistically_better = delta_low is not None and delta_low > 0.0
    selected_value = (
        "OUTSIDER_AGREE_ONLY"
        if hard_statistically_better
        else "FULL_WEIGHTED_MRF"
    )
    return {
        "parameter": "outsider_agreement",
        "selected": selected_value,
        "candidates": candidates,
        "difference": {
            "candidate_minus_baseline": "OUTSIDER_AGREE_ONLY - FULL_WEIGHTED_MRF",
            "pnl_delta": round(
                candidates[1]["net_pnl"] - candidates[0]["net_pnl"], 10
            ),
            "pnl_ci_low": delta_low,
            "pnl_ci_high": delta_high,
            "statistically_better": hard_statistically_better,
            "bootstrap_iterations": 1000,
        },
        "comparison": {
            "soft_models_agree": "FULL_WEIGHTED_MRF",
            "hard_consensus": "OUTSIDER_AGREE_ONLY",
        },
    }


def compare_mrf_application(
    observations: Sequence[MarketObservation],
    *,
    config: WeightedPolicyConfig | None = None,
    folds: Optional[Sequence[PurgedFold]] = None,
    evaluation_indices: Optional[Sequence[int]] = None,
    beta: float = 0.25,
    gamma: float = 0.25,
) -> dict[str, Any]:
    probability_config = replace(config or WeightedPolicyConfig(), mrf_beta=float(beta))
    probability_metrics = [
        evaluate_arm(
            observations,
            "FULL_WEIGHTED_MRF",
            config=probability_config,
            evaluation_indices=indices or None,
        )
        for indices in _tuning_indices(folds, evaluation_indices)
    ]
    probability = ArmMetrics(arm="FULL_WEIGHTED_MRF")
    for item in probability_metrics:
        probability.observations += item.observations
        probability.trades += item.trades
        probability.wins += item.wins
        probability.net_pnl += item.net_pnl
        probability.total_cost += item.total_cost
        probability.evaluations.extend(item.evaluations)
    probability.net_pnl = round(probability.net_pnl, 10)
    probability.total_cost = round(probability.total_cost, 10)
    stake_metrics = [
        evaluate_arm(
            observations,
            "FULL_WEIGHTED",
            config=config,
            mrf_stake_gamma=float(gamma),
            evaluation_indices=indices or None,
        )
        for indices in _tuning_indices(folds, evaluation_indices)
    ]
    stake = ArmMetrics(arm="FULL_WEIGHTED")
    for item in stake_metrics:
        stake.observations += item.observations
        stake.trades += item.trades
        stake.wins += item.wins
        stake.net_pnl += item.net_pnl
        stake.total_cost += item.total_cost
        stake.evaluations.extend(item.evaluations)
    stake.net_pnl = round(stake.net_pnl, 10)
    stake.total_cost = round(stake.total_cost, 10)
    return {
        "probability_adjustment": probability.as_dict(),
        "stake_adjustment": stake.as_dict(),
        "probability_net_pnl": probability.net_pnl,
        "stake_net_pnl": stake.net_pnl,
        "selected": (
            "probability_adjustment"
            if probability.net_pnl >= stake.net_pnl
            else "stake_adjustment"
        ),
        "folds": len(probability_metrics),
        "probability_stable_folds": sum(1 for item in probability_metrics if item.net_pnl > 0.0),
        "stake_stable_folds": sum(1 for item in stake_metrics if item.net_pnl > 0.0),
    }


def estimate_oof_standard_error(
    observations: Sequence[MarketObservation],
    predictions: Mapping[int, float],
    *,
    evaluation_indices: Optional[Sequence[int]] = None,
) -> Optional[float]:
    """Estimate calibration uncertainty from chronological OOF residuals."""
    allowed = (
        {int(index) for index in evaluation_indices}
        if evaluation_indices is not None
        else set(predictions)
    )
    residuals: list[float] = []
    for index, prediction in predictions.items():
        if index not in allowed:
            continue
        if index < 0 or index >= len(observations):
            continue
        outcome = observations[index].outcome_yes
        probability = clamp_probability(prediction)
        if outcome is None or probability is None:
            continue
        residuals.append(float(outcome) - probability)
    if len(residuals) < 2:
        return None
    estimate = float(np.std(np.asarray(residuals), ddof=1))
    return round(max(0.0, min(0.5, estimate)), 8)




def _effective_sizing_standard_error(
    configured: Optional[float],
    oof_estimate: Optional[float],
) -> float:
    """Prefer an explicit positive override, otherwise use OOF uncertainty."""
    try:
        configured_value = float(configured) if configured is not None else None
    except (TypeError, ValueError, OverflowError):
        configured_value = None
    if (
        configured_value is not None
        and np.isfinite(configured_value)
        and configured_value > 0.0
    ):
        return min(0.5, configured_value)
    try:
        oof_value = float(oof_estimate) if oof_estimate is not None else None
    except (TypeError, ValueError, OverflowError):
        oof_value = None
    if oof_value is not None and np.isfinite(oof_value) and oof_value >= 0.0:
        return min(0.5, oof_value)
    return 0.0


def parameter_sensitivity(
    observations: Sequence[MarketObservation],
    *,
    arm: str = "FULL_WEIGHTED_MRF",
    config: WeightedPolicyConfig | None = None,
    parameters: Sequence[str] = (
        "market_weight",
        "logreg_weight",
        "lgbm_weight",
        "mrf_beta",
        "fee_rate",
        "slippage_rate",
    ),
    deltas: Sequence[float] = (-0.20, -0.10, 0.10, 0.20),
    evaluation_indices: Optional[Sequence[int]] = None,
    sizing_mode: str = "FIXED",
    sizing_standard_error: float = 0.0,
    sizing_kelly_fraction: float = 0.025,
    sizing_base_bet_usdc: float = 1.0,
    sizing_cap_usdc: float = 3.0,
) -> tuple[dict[str, Any], ...]:
    """Evaluate bounded parameter perturbations on one fixed OOT sample."""
    base = config or WeightedPolicyConfig()
    rows: list[dict[str, Any]] = []
    for parameter in parameters:
        if not hasattr(base, parameter):
            continue
        try:
            baseline = float(getattr(base, parameter))
        except (TypeError, ValueError, OverflowError):
            continue
        for delta in deltas:
            try:
                perturbation = float(delta)
            except (TypeError, ValueError, OverflowError):
                continue
            scale_for_zero = {
                "market_weight": 0.05,
                "logreg_weight": 0.05,
                "lgbm_weight": 0.05,
                "mrf_beta": 1.0,
                "fee_rate": 0.01,
                "slippage_rate": 0.01,
            }.get(parameter, 1.0)
            value = (
                baseline * (1.0 + perturbation)
                if baseline != 0.0
                else perturbation * scale_for_zero
            )
            if parameter.endswith("_weight"):
                value = max(0.0, min(1.0, value))
            elif parameter == "mrf_beta":
                value = max(-2.0, min(2.0, value))
            else:
                value = max(0.0, value)
            metrics = evaluate_arm(
                observations,
                arm,
                config=replace(base, **{parameter: value}),
                evaluation_indices=evaluation_indices,
                sizing_mode=sizing_mode,
                sizing_standard_error=sizing_standard_error,
                sizing_kelly_fraction=sizing_kelly_fraction,
                sizing_base_bet_usdc=sizing_base_bet_usdc,
                sizing_cap_usdc=sizing_cap_usdc,
            )
            rows.append(
                {
                    "type": "PARAMETER_SENSITIVITY",
                    "arm": arm.upper(),
                    "parameter": parameter,
                    "delta": perturbation,
                    "baseline": baseline,
                    "value": value,
                    "trades": metrics.trades,
                    "net_pnl": metrics.net_pnl,
                    "win_rate": metrics.win_rate,
                    "brier": metrics.brier,
                    "log_loss": metrics.log_loss,
                }
            )
    return tuple(rows)


def stability_by_segment(
    observations: Sequence[MarketObservation],
    *,
    arm: str = "FULL_WEIGHTED_MRF",
    config: WeightedPolicyConfig | None = None,
    evaluation_indices: Optional[Sequence[int]] = None,
    sizing_mode: str = "FIXED",
    sizing_standard_error: float = 0.0,
    sizing_kelly_fraction: float = 0.025,
    sizing_base_bet_usdc: float = 1.0,
    sizing_cap_usdc: float = 3.0,
    stacker_predictions: Optional[Mapping[int, float]] = None,
    use_stacker_predictions: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Report OOT net PnL by asset, role, horizon and execution role."""
    allowed = (
        {int(index) for index in evaluation_indices}
        if evaluation_indices is not None
        else set(range(len(observations)))
    )
    dimensions = (
        "asset",
        "market_role",
        "phase",
        "asset_phase",
        "horizon",
        "execution_role",
        "consensus",
        "week",
    )
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        groups: dict[str, list[int]] = {}
        for index, item in enumerate(observations):
            if index not in allowed:
                continue
            if dimension == "asset":
                key = item.asset.strip().upper() or "UNKNOWN"
            elif dimension == "market_role":
                key = _observation_role(item)
            elif dimension == "phase":
                key = str(item.phase or "UNKNOWN").strip().upper() or "UNKNOWN"
            elif dimension == "asset_phase":
                key = str(item.asset_phase or "UNKNOWN").strip().upper() or "UNKNOWN"
            elif dimension == "horizon":
                key = normalize_horizon(item.horizon) or "UNKNOWN"
            elif dimension == "consensus":
                agreement = _models_agree(item)
                key = (
                    "AGREE"
                    if agreement is True
                    else "DISAGREE"
                    if agreement is False
                    else "UNKNOWN"
                )
            elif dimension == "week":
                key = item.timestamp.strftime("%G-W%V")
            else:
                key = str(item.execution_role or "UNKNOWN").strip().upper() or "UNKNOWN"
            groups.setdefault(key, []).append(index)
        for key, indices in sorted(groups.items()):
            metrics = evaluate_arm(
                observations,
                arm,
                config=config,
                evaluation_indices=indices,
                stacker_predictions=stacker_predictions,
                use_stacker_predictions=use_stacker_predictions,
                sizing_mode=sizing_mode,
                sizing_standard_error=sizing_standard_error,
                sizing_kelly_fraction=sizing_kelly_fraction,
                sizing_base_bet_usdc=sizing_base_bet_usdc,
                sizing_cap_usdc=sizing_cap_usdc,
            )
            ci_low, ci_high = cluster_bootstrap_ci(
                metrics.evaluations,
                iterations=1000,
                seed=20260901,
            )
            rows.append(
                {
                    "dimension": dimension,
                    "segment": key,
                    "arm": arm.upper(),
                    "observations": metrics.observations,
                    "trades": metrics.trades,
                    "net_pnl": metrics.net_pnl,
                    "roi": metrics.roi,
                    "win_rate": metrics.win_rate,
                    "pnl_ci_low": ci_low,
                    "pnl_ci_high": ci_high,
                }
            )
    return tuple(rows)


def evaluate_sizing_steps(
    observations: Sequence[MarketObservation],
    *,
    arm: str = "FULL_WEIGHTED_MRF",
    config: WeightedPolicyConfig | None = None,
    levels: Sequence[float] = (1.0, 1.5, 2.0, 3.0),
    min_net_ev: float = 0.0,
    evaluation_indices: Optional[Sequence[int]] = None,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 20260901,
    stacker_predictions: Optional[Mapping[int, float]] = None,
    use_stacker_predictions: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Compare fixed stake levels on one identical OOT sample."""
    rows: list[dict[str, Any]] = []
    for level in levels:
        try:
            stake = max(0.0, float(level))
        except (TypeError, ValueError, OverflowError):
            continue
        metrics = evaluate_arm(
            observations,
            arm,
            config=config,
            min_net_ev=min_net_ev,
            sizing_mode="FIXED",
            sizing_base_bet_usdc=stake,
            stacker_predictions=stacker_predictions,
            use_stacker_predictions=use_stacker_predictions,
            evaluation_indices=evaluation_indices,
        )
        ci_low, ci_high = cluster_bootstrap_ci(
            metrics.evaluations,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        rows.append(
            {
                "type": "SIZING_STEP",
                "arm": arm.upper(),
                "stake_usdc": stake,
                "observations": metrics.observations,
                "trades": metrics.trades,
                "net_pnl": metrics.net_pnl,
                "win_rate": metrics.win_rate,
                "brier": metrics.brier,
                "log_loss": metrics.log_loss,
                "pnl_ci_low": ci_low,
                "pnl_ci_high": ci_high,
            }
        )
    return tuple(rows)


def _max_drawdown(evaluations: Sequence[TradeEvaluation]) -> float:
    """Return the worst cumulative PnL drawdown for chronological fills."""
    balance = 0.0
    peak = 0.0
    drawdown = 0.0
    for item in sorted(evaluations, key=lambda value: (value.timestamp, value.market_id)):
        balance += float(item.pnl)
        peak = max(peak, balance)
        drawdown = min(drawdown, balance - peak)
    return round(drawdown, 10)


def compare_kelly_fractions(
    observations: Sequence[MarketObservation],
    *,
    arm: str = "FULL_WEIGHTED_MRF",
    fractions: Sequence[float] = (0.025, 0.05, 0.10),
    config: WeightedPolicyConfig | None = None,
    standard_error: float = 0.0,
    min_net_ev: float = 0.0,
    evaluation_indices: Optional[Sequence[int]] = None,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 20260901,
    stacker_predictions: Optional[Mapping[int, float]] = None,
    use_stacker_predictions: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Compare 2.5/5/10% lower-bound Kelly on one fixed OOT sample."""
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        try:
            value = max(0.0, min(1.0, float(fraction)))
        except (TypeError, ValueError, OverflowError):
            continue
        metrics = evaluate_arm(
            observations,
            arm,
            config=config,
            min_net_ev=min_net_ev,
            sizing_mode="LOWER_BOUND_KELLY",
            sizing_standard_error=standard_error,
            sizing_kelly_fraction=value,
            stacker_predictions=stacker_predictions,
            use_stacker_predictions=use_stacker_predictions,
            evaluation_indices=evaluation_indices,
        )
        ci_low, ci_high = cluster_bootstrap_ci(
            metrics.evaluations,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        max_drawdown = _max_drawdown(metrics.evaluations)
        risk_denominator = abs(max_drawdown)
        drawdown_adjusted = (
            metrics.net_pnl / risk_denominator
            if risk_denominator > 0.0
            else metrics.net_pnl
        )
        rows.append(
            {
                "type": "KELLY_FRACTION",
                "arm": arm.upper(),
                "fraction": value,
                "fraction_percent": round(value * 100.0, 4),
                "observations": metrics.observations,
                "trades": metrics.trades,
                "net_pnl": metrics.net_pnl,
                "win_rate": metrics.win_rate,
                "brier": metrics.brier,
                "log_loss": metrics.log_loss,
                "pnl_ci_low": ci_low,
                "pnl_ci_high": ci_high,
                "max_drawdown": max_drawdown,
                "drawdown_adjusted_pnl": round(drawdown_adjusted, 10),
            }
        )
    return tuple(rows)


def cluster_bootstrap_ci(
    evaluations: Sequence[TradeEvaluation],
    *,
    iterations: int = 1000,
    seed: int = 20260901,
    alpha: float = 0.05,
) -> tuple[Optional[float], Optional[float]]:
    """Return a cluster bootstrap CI for aggregate net PnL."""
    if not evaluations:
        return None, None
    grouped: dict[str, list[float]] = {}
    for item in evaluations:
        timestamp = item.timestamp.astimezone(timezone.utc)
        cluster = f"{item.market_id}|{timestamp.date().isoformat()}"
        grouped.setdefault(cluster, []).append(item.pnl)
    keys = list(grouped)
    rng = np.random.default_rng(seed)
    samples = [
        sum(sum(grouped[keys[i]]) for i in rng.integers(0, len(keys), len(keys)))
        for _ in range(max(1, int(iterations)))
    ]
    return (
        round(float(np.quantile(samples, alpha / 2.0)), 10),
        round(float(np.quantile(samples, 1.0 - alpha / 2.0)), 10),
    )


def fingerprint_observations(
    observations: Sequence[MarketObservation],
) -> str:
    """Return a stable SHA-256 fingerprint for an ordered dataset."""
    payload = [
        item.as_dict()
        for item in sorted(
            observations,
            key=lambda item: (
                item.timestamp,
                item.market_id,
                normalize_horizon(item.horizon),
            ),
        )
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_policy_artifact_from_benchmark(
    observations: Sequence[MarketObservation],
    report: BenchmarkReport,
    *,
    version: str,
    policy_config: Optional[WeightedPolicyConfig] = None,
    thresholds: Optional[Mapping[str, Any]] = None,
    source_report_hash: Optional[str] = None,
):
    """Create an immutable policy artifact tied to one benchmark dataset."""
    from polyflip.trading.policy_artifact import create_policy_artifact

    has_horizon_labels = any(
        bool(str(item.horizon or "").strip()) for item in observations
    )
    source = (
        filter_fixed_horizons(observations)
        if has_horizon_labels
        else tuple(observations)
    )
    ordered = deduplicate_observations(source)
    fingerprint = fingerprint_observations(ordered)
    if report.dataset_fingerprint and report.dataset_fingerprint != fingerprint:
        raise ValueError("benchmark report does not match supplied observations")
    timestamps = [item.timestamp for item in ordered]
    training_window = {
        "first_timestamp": min(timestamps).isoformat() if timestamps else None,
        "last_timestamp": max(timestamps).isoformat() if timestamps else None,
        "observations": len(ordered),
        "resolved_observations": sum(
            1 for item in ordered if item.outcome_yes is not None
        ),
        "folds": len(report.folds),
        "oof_standard_error": report.oof_standard_error,
    }
    return create_policy_artifact(
        version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        training_window=training_window,
        stacker=report.stacker,
        policy_config=policy_config or WeightedPolicyConfig(),
        thresholds=dict(thresholds or {}),
        hierarchical_stacker=(
            report.hierarchical_stacker.as_dict()
            if report.hierarchical_stacker is not None
            else None
        ),
        source_report_hash=source_report_hash,
        dataset_fingerprint=fingerprint,
    )


def benchmark(
    observations: Sequence[MarketObservation],
    *,
    config: BenchmarkConfig | None = None,
    arms: Sequence[str] = (
        "MARKET_ONLY",
        "LEGACY",
        "MARKET_LOGREG",
        "MARKET_LGBM",
        "FULL_WEIGHTED",
        "FULL_WEIGHTED_MRF",
        "OUTSIDER_AGREE_ONLY",
    ),
) -> BenchmarkReport:
    """Run purged walk-forward stacker training plus comparison arms."""
    cfg = config or BenchmarkConfig()
    raw_count = len(observations)
    has_horizon_labels = any(
        bool(str(item.horizon or "").strip()) for item in observations
    )
    source_observations = (
        filter_fixed_horizons(observations)
        if has_horizon_labels
        else tuple(observations)
    )
    ordered = deduplicate_observations(source_observations)
    resolved = tuple(item for item in ordered if item.outcome_yes is not None)
    dataset_fp = fingerprint_observations(ordered)
    folds = purged_walk_forward_folds(
        ordered,
        train_min_rows=cfg.train_min_rows,
        test_size=cfg.test_size,
        purge_gap=cfg.purge_gap,
    )
    stacker_model = None
    oof: dict[int, float] = {}
    deployment_oof: dict[int, float] = {}
    for fold in folds:
        train_rows = [ordered[i] for i in fold.train_indices]
        try:
            model = fit_ridge_logistic_stacker(
                train_rows,
                ridge_lambda=cfg.ridge_lambda,
                coefficient_bound=cfg.coefficient_bound,
            )
        except ValueError:
            # Historical rows can lack a quote/model probability.  Keep the
            # fold in the audit trail but do not let one sparse train window
            # invalidate the remaining OOT folds.
            continue
        fold_hierarchical = None
        try:
            fold_hierarchical = fit_hierarchical_stackers(
                train_rows,
                global_model=model,
                min_segment_rows=cfg.hierarchical_min_segment_rows,
                shrinkage=cfg.hierarchical_shrinkage,
                ridge_lambda=cfg.ridge_lambda,
                coefficient_bound=cfg.coefficient_bound,
            )
        except ValueError:
            # The global fold model remains a valid fallback when a segment
            # cannot be fitted from that training window.
            pass
        for i in fold.test_indices:
            p = model.predict_one(ordered[i])
            if p is not None:
                oof[i] = p
            deployment_prediction = (
                fold_hierarchical.predict_one(ordered[i])
                if fold_hierarchical is not None
                else p
            )
            if deployment_prediction is not None:
                deployment_oof[i] = deployment_prediction
    oot_indices = tuple(
        sorted({index for fold in folds for index in fold.test_indices})
    )
    oof_standard_error = estimate_oof_standard_error(
        ordered,
        deployment_oof,
        evaluation_indices=oot_indices,
    )
    if folds:
        try:
            stacker_model = fit_ridge_logistic_stacker(
                resolved,
                ridge_lambda=cfg.ridge_lambda,
                coefficient_bound=cfg.coefficient_bound,
            )
        except ValueError:
            stacker_model = None
    hierarchical_model = None
    try:
        hierarchical_model = fit_hierarchical_stackers(
            resolved,
            global_model=stacker_model,
            min_segment_rows=cfg.hierarchical_min_segment_rows,
            shrinkage=cfg.hierarchical_shrinkage,
            ridge_lambda=cfg.ridge_lambda,
            coefficient_bound=cfg.coefficient_bound,
        )
    except ValueError:
        hierarchical_model = None
    # Keep ordinary OOT arms free of any full-sample fitted coefficients. The
    # deployable artifact is evaluated with fold-held-out predictions below;
    # its full-sample hierarchical coefficients are reserved for final export.
    base_evaluation_config = replace(
        cfg.policy_config,
        stacker_feature_names=(),
        stacker_coefficients=(),
        stacker_segment_models=(),
    )
    deployment_policy_config = base_evaluation_config
    if hierarchical_model is not None:
        deployment_policy_config = replace(
            base_evaluation_config,
            stacker_feature_names=hierarchical_model.global_model.feature_names,
            stacker_coefficients=hierarchical_model.global_model.coefficients,
            stacker_segment_models=tuple(
                sorted(
                    (key, model.coefficients)
                    for key, model in hierarchical_model.segment_models.items()
                )
            ),
        )
    effective_sizing_standard_error = _effective_sizing_standard_error(
        cfg.sizing_standard_error,
        oof_standard_error,
    )
    sizing_kwargs = {
        "sizing_mode": cfg.sizing_mode,
        "sizing_standard_error": effective_sizing_standard_error,
        "sizing_kelly_fraction": cfg.sizing_kelly_fraction,
        "sizing_base_bet_usdc": cfg.sizing_base_bet_usdc,
        "sizing_cap_usdc": cfg.sizing_cap_usdc,
    }
    results: list[ArmMetrics] = []
    for arm in arms:
        arm_name = arm.upper()
        use_deployment_oof = arm_name in {"FULL_WEIGHTED_MRF", "OUTSIDER_AGREE_ONLY"}
        result = evaluate_arm(
            ordered,
            arm,
            config=(deployment_policy_config if use_deployment_oof else base_evaluation_config),
            min_net_ev=cfg.min_net_ev,
            stacker=stacker_model,
            stacker_predictions=(
                deployment_oof
                if use_deployment_oof
                else oof
                if arm_name == "STACKER"
                else None
            ),
            use_stacker_predictions=use_deployment_oof,
            evaluation_indices=oot_indices,
            **sizing_kwargs,
        )
        result.pnl_ci_low, result.pnl_ci_high = cluster_bootstrap_ci(
            result.evaluations,
            iterations=cfg.bootstrap_iterations,
            seed=cfg.bootstrap_seed,
        )
        results.append(result)
    if stacker_model is not None:
        result = evaluate_arm(
            ordered,
            "STACKER",
            config=deployment_policy_config,
            min_net_ev=cfg.min_net_ev,
            evaluation_indices=oot_indices,
            stacker=stacker_model,
            stacker_predictions=deployment_oof,
            **sizing_kwargs,
        )
        result.pnl_ci_low, result.pnl_ci_high = cluster_bootstrap_ci(
            result.evaluations,
            iterations=cfg.bootstrap_iterations,
            seed=cfg.bootstrap_seed,
        )
        results.append(result)
    sensitivity = []
    for threshold in cfg.candidate_min_net_ev:
        for arm in ("FULL_WEIGHTED_MRF", "OUTSIDER_AGREE_ONLY"):
            result = evaluate_arm(
                ordered,
                arm,
                config=base_evaluation_config,
                min_net_ev=float(threshold),
                evaluation_indices=oot_indices,
                **sizing_kwargs,
            )
            sensitivity.append(
                {
                    "arm": arm,
                    "min_net_ev": float(threshold),
                    "trades": result.trades,
                    "net_pnl": round(result.net_pnl, 10),
                    "win_rate": result.win_rate,
                }
            )
    sensitivity.extend(
        parameter_sensitivity(
            ordered,
            arm="FULL_WEIGHTED_MRF",
            config=base_evaluation_config,
            evaluation_indices=oot_indices,
            **sizing_kwargs,
        )
    )
    stability = stability_by_segment(
        ordered,
        arm="FULL_WEIGHTED_MRF",
        config=deployment_policy_config,
        evaluation_indices=oot_indices,
        stacker_predictions=deployment_oof,
        use_stacker_predictions=True,
        **sizing_kwargs,
    )
    sizing_steps = evaluate_sizing_steps(
        ordered,
        arm="FULL_WEIGHTED_MRF",
        config=deployment_policy_config,
        min_net_ev=cfg.min_net_ev,
        evaluation_indices=oot_indices,
        stacker_predictions=deployment_oof,
        use_stacker_predictions=True,
        bootstrap_iterations=cfg.bootstrap_iterations,
        bootstrap_seed=cfg.bootstrap_seed,
    )
    tuning_results: list[dict[str, Any]] = []
    mrf_beta_result = optimize_mrf_beta(
        ordered,
        candidate_values=cfg.candidate_mrf_beta,
        config=base_evaluation_config,
        folds=folds,
        evaluation_indices=None if folds else None,
        min_net_ev=cfg.min_net_ev,
    )
    tuning_results.append(mrf_beta_result.as_dict())
    selected_beta = (
        float(mrf_beta_result.selected)
        if mrf_beta_result.selected is not None
        else float(cfg.policy_config.mrf_beta)
    )
    tuning_results.append(
        {
            "parameter": "mrf_application",
            "result": compare_mrf_application(
                ordered,
                config=base_evaluation_config,
                folds=folds,
                evaluation_indices=None if folds else None,
                beta=selected_beta,
            ),
        }
    )
    tuning_results.append(
        compare_outsider_agreement(
            ordered,
            config=deployment_policy_config,
            folds=folds,
            evaluation_indices=None if folds else None,
            min_net_ev=cfg.min_net_ev,
            stacker_predictions=deployment_oof,
            use_stacker_predictions=True,
        )
    )
    for role in ("FAVORITE", "OUTSIDER"):
        tuning_results.append(
            optimize_min_net_ev(
                ordered,
                role=role,
                arm="FULL_WEIGHTED_MRF",
                candidate_values=cfg.candidate_min_net_ev,
                config=deployment_policy_config,
                folds=folds,
                evaluation_indices=None if folds else None,
                stacker_predictions=deployment_oof,
                use_stacker_predictions=True,
            ).as_dict()
        )
        tuning_results.append(
            optimize_price_cap(
                ordered,
                role=role,
                arm="FULL_WEIGHTED_MRF",
                candidate_values=(
                    cfg.candidate_favorite_price_caps
                    if role == "FAVORITE"
                    else cfg.candidate_price_caps
                ),
                config=deployment_policy_config,
                folds=folds,
                evaluation_indices=None if folds else None,
                stacker_predictions=deployment_oof,
                use_stacker_predictions=True,
            ).as_dict()
        )
        tuning_results.append(
            optimize_time_window(
                ordered,
                role=role,
                windows=cfg.candidate_time_windows,
                arm="FULL_WEIGHTED_MRF",
                config=deployment_policy_config,
                folds=folds,
                evaluation_indices=None if folds else None,
                stacker_predictions=deployment_oof,
                use_stacker_predictions=True,
            ).as_dict()
        )
    kelly_fractions = compare_kelly_fractions(
        ordered,
        arm="FULL_WEIGHTED_MRF",
        config=deployment_policy_config,
        stacker_predictions=deployment_oof,
        use_stacker_predictions=True,
        standard_error=(
            oof_standard_error
            if oof_standard_error is not None
            else cfg.sizing_standard_error
        ),
        min_net_ev=cfg.min_net_ev,
        evaluation_indices=oot_indices,
        bootstrap_iterations=cfg.bootstrap_iterations,
        bootstrap_seed=cfg.bootstrap_seed,
    )
    return BenchmarkReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        observations=len(ordered),
        resolved_observations=len(resolved),
        folds=folds,
        stacker=stacker_model,
        arms=tuple(results),
        sensitivity=tuple(sensitivity),
        hierarchical_stacker=hierarchical_model,
        duplicate_rows_removed=max(0, raw_count - len(ordered)),
        dataset_fingerprint=dataset_fp,
        oof_standard_error=oof_standard_error,
        stability=stability,
        sizing_steps=sizing_steps,
        kelly_fractions=kelly_fractions,
        tuning=tuple(tuning_results),
    )


@dataclass(frozen=True)
class HierarchicalStacker:
    global_model: StackerModel
    segment_models: Mapping[str, StackerModel]
    min_segment_rows: int
    shrinkage: float

    def predict_one(self, observation: MarketObservation) -> Optional[float]:
        model = self.segment_models.get(
            observation_segment_key(observation), self.global_model
        )
        return model.predict_one(observation)

    def as_dict(self) -> dict[str, Any]:
        return {
            "global": self.global_model.as_dict(),
            "segments": {
                key: model.as_dict()
                for key, model in self.segment_models.items()
            },
            "min_segment_rows": self.min_segment_rows,
            "shrinkage": self.shrinkage,
        }


def fit_hierarchical_stackers(
    observations: Sequence[MarketObservation],
    *,
    global_model: Optional[StackerModel] = None,
    min_segment_rows: int = 300,
    shrinkage: float = 300.0,
    ridge_lambda: float = 1.0,
    coefficient_bound: float = 5.0,
) -> HierarchicalStacker:
    """Fit segment models and shrink them toward the global model."""
    resolved = tuple(
        item
        for item in observations
        if item.outcome_yes is not None and _stacker_features(item) is not None
    )
    if global_model is None:
        global_model = fit_ridge_logistic_stacker(
            resolved,
            ridge_lambda=ridge_lambda,
            coefficient_bound=coefficient_bound,
        )
    minimum = max(1, int(min_segment_rows))
    shrink = max(0.0, float(shrinkage))
    grouped: dict[str, list[MarketObservation]] = {}
    for item in resolved:
        grouped.setdefault(observation_segment_key(item), []).append(item)
    segment_models: dict[str, StackerModel] = {}
    for key, rows in grouped.items():
        independent_markets = {_independent_market_key(item) for item in rows}
        if len(independent_markets) < minimum:
            continue
        fitted = fit_ridge_logistic_stacker(
            rows,
            ridge_lambda=ridge_lambda,
            coefficient_bound=coefficient_bound,
        )
        alpha = len(rows) / (len(rows) + shrink) if shrink > 0.0 else 1.0
        coefficients = tuple(
            round(
                global_value * (1.0 - alpha) + segment_value * alpha,
                10,
            )
            for global_value, segment_value in zip(
                global_model.coefficients, fitted.coefficients
            )
        )
        segment_models[key] = StackerModel(
            feature_names=global_model.feature_names,
            coefficients=coefficients,
            training_rows=len(rows),
            ridge_lambda=fitted.ridge_lambda,
            coefficient_bound=fitted.coefficient_bound,
            training_markets=len(independent_markets),
        )
    return HierarchicalStacker(
        global_model=global_model,
        segment_models=segment_models,
        min_segment_rows=minimum,
        shrinkage=shrink,
    )


def observation_key(observation: MarketObservation) -> tuple[str, str]:
    market_id = observation.market_id or (
        f"{observation.asset}|{observation.timestamp.isoformat()}"
    )
    return market_id, normalize_horizon(observation.horizon)


def deduplicate_observations(
    observations: Sequence[MarketObservation],
) -> tuple[MarketObservation, ...]:
    """Keep the first chronological row for each market and horizon."""
    selected: dict[tuple[str, str], MarketObservation] = {}
    for item in observations:
        key = observation_key(item)
        previous = selected.get(key)
        if previous is None or item.timestamp < previous.timestamp:
            selected[key] = item
    return tuple(sorted(selected.values(), key=lambda item: item.timestamp))


def filter_fixed_horizons(
    observations: Sequence[MarketObservation],
) -> tuple[MarketObservation, ...]:
    """Return only the fixed 10M/5M/2M benchmark horizons."""
    return tuple(
        item
        for item in observations
        if normalize_horizon(item.horizon) in FIXED_HORIZONS
    )
