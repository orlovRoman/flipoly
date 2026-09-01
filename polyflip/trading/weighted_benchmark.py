"""Cost-aware offline evaluation for the weighted trading policy.

The module is pure apart from numpy and accepts exported database rows.  It
shares the live policy's probability, side-selection, fee and slippage code so
the benchmark cannot silently use a different EV definition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import log
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from polyflip.trading.weighted_policy import (
    BUY_NO,
    BUY_YES,
    WeightedPolicyConfig,
    clamp_probability,
    logit,
    score_weighted_probability,
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
    observed_cost_per_share: Optional[float] = None
    group: Optional[str] = None
    horizon: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MarketObservation":
        yes_ask = raw.get("yes_ask", raw.get("candidate_ask"))
        no_ask = raw.get("no_ask")
        p_market = raw.get("p_market_yes", raw.get("weighted_p_market_yes"))
        if p_market is None and yes_ask is not None and no_ask is not None:
            try:
                yes, no = float(yes_ask), float(no_ask)
                if yes > 0.0 and no > 0.0:
                    p_market = yes / (yes + no)
            except (TypeError, ValueError):
                pass
        raw_evidence = raw.get("mrf_evidence", raw.get("weighted_mrf_evidence"))
        observed_cost = raw.get("observed_cost_per_share")
        if observed_cost is None:
            observed_cost = raw.get("observed_fee_per_share")
        return cls(
            market_id=str(raw.get("market_id", raw.get("condition_id", ""))),
            timestamp=_dt(raw.get("timestamp", raw.get("created_at"))),
            asset=str(raw.get("asset", "")),
            yes_ask=_probability(yes_ask),
            no_ask=_probability(no_ask),
            outcome_yes=_outcome_yes(
                raw.get(
                    "outcome_yes",
                    raw.get("final_outcome", raw.get("settlement_outcome")),
                )
            ),
            p_market_yes=_probability(p_market),
            p_logreg_yes=_probability(
                raw.get(
                    "p_logreg_yes",
                    raw.get("weighted_p_logreg_yes", raw.get("p_logreg_win")),
                )
            ),
            p_lgbm_yes=_probability(
                raw.get("p_lgbm_yes", raw.get("weighted_p_lgbm_yes"))
            ),
            p_legacy_yes=_probability(raw.get("p_legacy_yes")),
            mrf_evidence=_evidence(raw_evidence),
            spread=max(0.0, float(raw.get("spread", 0.0) or 0.0)),
            fee_rate=(
                float(raw["fee_rate"]) if raw.get("fee_rate") is not None else None
            ),
            fee_exponent=(
                float(raw["fee_exponent"])
                if raw.get("fee_exponent") is not None
                else None
            ),
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
            observed_cost_per_share=(
                max(0.0, float(observed_cost))
                if observed_cost is not None
                else None
            ),
            group=str(raw.get("group", raw.get("asset", "")) or ""),
            horizon=normalize_horizon(raw.get("horizon", raw.get("market_horizon", ""))),
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
    candidate_min_net_ev: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05)
    hierarchical_min_segment_rows: int = 300
    hierarchical_shrinkage: float = 300.0


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
            "ridge_lambda": self.ridge_lambda,
            "coefficient_bound": self.coefficient_bound,
        }


@dataclass(frozen=True)
class TradeEvaluation:
    market_id: str
    asset: str
    side: str
    ask: float
    p_win: float
    outcome_yes: bool
    pnl: float
    cost_per_share: float
    group: str


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

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["win_rate"] = self.win_rate
        result["mean_pnl"] = self.mean_pnl
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
        }


def purged_walk_forward_folds(
    observations: Sequence[MarketObservation],
    *,
    train_min_rows: int = 300,
    test_size: int = 100,
    purge_gap: int = 0,
) -> tuple[PurgedFold, ...]:
    """Build chronological folds with a purge gap immediately before test."""
    order = sorted(range(len(observations)), key=lambda i: observations[i].timestamp)
    train_min_rows, test_size = max(1, int(train_min_rows)), max(1, int(test_size))
    purge_gap = max(0, int(purge_gap))
    folds: list[PurgedFold] = []
    test_start = train_min_rows + purge_gap
    while test_start < len(order):
        train_end = test_start - purge_gap
        test_end = min(len(order), test_start + test_size)
        folds.append(
            PurgedFold(
                train_indices=tuple(order[:train_end]),
                test_indices=tuple(order[test_start:test_end]),
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


def observation_segment_key(observation: MarketObservation) -> str:
    raw_role = str(observation.market_role or "UNKNOWN").strip().upper() or "UNKNOWN"
    role = "OUTSIDER" if _is_outsider_role(raw_role) else raw_role
    agreement = _models_agree(observation)
    agreement_label = "AGREE" if agreement is True else ("DISAGREE" if agreement is False else "UNKNOWN")
    return f"{observation.asset.strip().upper() or 'UNKNOWN'}|{role}|{agreement_label}"

def _stacker_features(observation: MarketObservation) -> Optional[tuple[float, ...]]:
    if observation.p_market_yes is None:
        return None
    market_logit = logit(observation.p_market_yes)
    agreement = _models_agree(observation)
    role_outsider = 1.0 if _is_outsider_role(observation.market_role) else 0.0
    models_agree = 1.0 if agreement is True else 0.0
    return (
        1.0,
        market_logit,
        logit(observation.p_logreg_yes) - market_logit
        if observation.p_logreg_yes is not None
        else 0.0,
        logit(observation.p_lgbm_yes) - market_logit
        if observation.p_lgbm_yes is not None
        else 0.0,
        float(observation.mrf_evidence or 0.0),
        role_outsider,
        models_agree,
        role_outsider * models_agree,
    )


def fit_ridge_logistic_stacker(
    observations: Sequence[MarketObservation],
    *,
    ridge_lambda: float = 1.0,
    coefficient_bound: float = 5.0,
    max_iter: int = 100,
) -> StackerModel:
    """Fit a bounded ridge-logistic stacker using only supplied rows."""
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
    penalty = max(0.0, float(ridge_lambda))
    bound = max(0.1, float(coefficient_bound))
    for _ in range(max(1, int(max_iter))):
        logits = np.clip(x @ beta, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
        gradient = x.T @ (probabilities - y) + penalty * beta
        hessian = (x.T * weights) @ x
        ridge = np.eye(x.shape[1], dtype=float) * penalty
        ridge[0, 0] = 0.0
        hessian += ridge + np.eye(x.shape[1]) * 1e-8
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        updated = np.clip(beta - step, -bound, bound)
        if np.max(np.abs(updated - beta)) < 1e-7:
            beta = updated
            break
        beta = updated
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
        ),
        coefficients=tuple(round(float(value), 10) for value in beta),
        training_rows=len(rows),
        ridge_lambda=penalty,
        coefficient_bound=bound,
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
    if arm in {"FULL_WEIGHTED_MRF", "WEIGHTED"}:
        return (
            observation.p_market_yes,
            observation.p_logreg_yes,
            observation.p_lgbm_yes,
            None,
        )
    if arm == "OUTSIDER_AGREE":
        if observation.market_role != "OUTSIDER":
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


def evaluate_arm(
    observations: Sequence[MarketObservation],
    arm: str,
    *,
    config: WeightedPolicyConfig | None = None,
    min_net_ev: float = 0.0,
    stacker: Optional[StackerModel] = None,
    stacker_predictions: Optional[Mapping[int, float]] = None,
) -> ArmMetrics:
    """Evaluate one arm using net realized PnL per one-share trade."""
    base = config or WeightedPolicyConfig()
    arm_name = arm.upper()
    metrics = ArmMetrics(arm=arm_name)
    brier, logloss = [], []
    for index, observation in enumerate(observations):
        if observation.outcome_yes is None:
            continue
        metrics.observations += 1
        if arm_name == "STACKER" and stacker_predictions is not None:
            p = stacker_predictions.get(index)
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
                    if arm_name in {"FULL_WEIGHTED_MRF", "WEIGHTED", "OUTSIDER_AGREE"}
                    else None
                ),
            )
            p_final = scored.p_final_yes
        outcome = 1.0 if observation.outcome_yes else 0.0
        brier.append((p_final - outcome) ** 2)
        logloss.append(-log(max(1e-12, p_final if observation.outcome_yes else 1.0 - p_final)))
        mrf = (
            observation.mrf_evidence
            if arm_name in {"FULL_WEIGHTED_MRF", "WEIGHTED", "OUTSIDER_AGREE"}
            else None
        )
        selected = select_weighted_side(
            p_market_yes=p_market if p_direct is None else p_final,
            p_logreg_yes=p_logreg if p_direct is None else None,
            p_lgbm_yes=p_lgbm if p_direct is None else None,
            yes_ask=observation.yes_ask,
            no_ask=observation.no_ask,
            config=policy,
            mrf_evidence=mrf,
            min_net_ev=min_net_ev,
            fee_source=observation.fee_source,
            spread=observation.spread,
            mrf_extreme_veto_threshold=policy.mrf_extreme_veto_threshold,
        )
        if selected.selected is None:
            continue
        quote = selected.selected
        won = observation.outcome_yes if quote.side == BUY_YES else not observation.outcome_yes
        cost = (
            observation.observed_cost_per_share
            if observation.observed_cost_per_share is not None
            else quote.cost.total_per_share
        )
        pnl = (1.0 if won else 0.0) - quote.ask - cost
        metrics.evaluations.append(
            TradeEvaluation(
                market_id=observation.market_id,
                asset=observation.asset,
                side=quote.side,
                ask=quote.ask,
                p_win=quote.p_win,
                outcome_yes=observation.outcome_yes,
                pnl=round(pnl, 10),
                cost_per_share=round(cost, 10),
                group=observation.group or observation.asset or observation.market_id,
            )
        )
        metrics.trades += 1
        metrics.wins += int(won)
        metrics.net_pnl += pnl
        metrics.total_cost += cost
    metrics.net_pnl = round(metrics.net_pnl, 10)
    metrics.total_cost = round(metrics.total_cost, 10)
    metrics.brier = round(float(np.mean(brier)), 10) if brier else None
    metrics.log_loss = round(float(np.mean(logloss)), 10) if logloss else None
    return metrics


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
        grouped.setdefault(item.group or item.market_id, []).append(item.pnl)
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


def benchmark(
    observations: Sequence[MarketObservation],
    *,
    config: BenchmarkConfig | None = None,
    arms: Sequence[str] = (
        "MARKET_ONLY",
        "LEGACY",
        "MARKET_LOGREG",
        "MARKET_LGBM",
        "FULL_WEIGHTED_MRF",
        "OUTSIDER_AGREE",
    ),
) -> BenchmarkReport:
    """Run purged walk-forward stacker training plus comparison arms."""
    cfg = config or BenchmarkConfig()
    raw_count = len(observations)
    ordered = deduplicate_observations(observations)
    resolved = tuple(item for item in ordered if item.outcome_yes is not None)
    folds = purged_walk_forward_folds(
        ordered,
        train_min_rows=cfg.train_min_rows,
        test_size=cfg.test_size,
        purge_gap=cfg.purge_gap,
    )
    stacker_model = None
    oof: dict[int, float] = {}
    for fold in folds:
        try:
            model = fit_ridge_logistic_stacker(
                [ordered[i] for i in fold.train_indices],
                ridge_lambda=cfg.ridge_lambda,
                coefficient_bound=cfg.coefficient_bound,
            )
        except ValueError:
            # Historical rows can lack a quote/model probability.  Keep the
            # fold in the audit trail but do not let one sparse train window
            # invalidate the remaining OOT folds.
            continue
        for i in fold.test_indices:
            p = model.predict_one(ordered[i])
            if p is not None:
                oof[i] = p
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
    results: list[ArmMetrics] = []
    for arm in arms:
        result = evaluate_arm(
            ordered,
            arm,
            config=cfg.policy_config,
            min_net_ev=cfg.min_net_ev,
            stacker=stacker_model,
            stacker_predictions=oof if arm.upper() == "STACKER" else None,
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
            config=cfg.policy_config,
            min_net_ev=cfg.min_net_ev,
            stacker=stacker_model,
            stacker_predictions=oof,
        )
        result.pnl_ci_low, result.pnl_ci_high = cluster_bootstrap_ci(
            result.evaluations,
            iterations=cfg.bootstrap_iterations,
            seed=cfg.bootstrap_seed,
        )
        results.append(result)
    sensitivity = []
    for threshold in cfg.candidate_min_net_ev:
        for arm in ("FULL_WEIGHTED_MRF", "OUTSIDER_AGREE"):
            result = evaluate_arm(
                ordered,
                arm,
                config=cfg.policy_config,
                min_net_ev=float(threshold),
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
        if len(rows) < minimum:
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
    return market_id, observation.horizon.strip().upper()


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
