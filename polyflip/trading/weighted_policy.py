"""Pure weighted trading policy for combining market and model probabilities.

The legacy decision path historically treated LogReg and LightGBM as hard
votes.  This module keeps the new policy deliberately small and side-effect
free so it can be used by the live evaluator, backtests, and shadow analysis
with exactly the same maths.

All probabilities are expressed as ``P(YES wins)``.  A BUY_NO candidate uses
``1 - P(YES)``.  Costs are expressed per share, in USDC, which keeps the
expected-value calculation in the same units as a binary contract payout.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import exp, isfinite, log
from typing import Any, Optional


BUY_YES = "BUY_YES"
BUY_NO = "BUY_NO"


def clamp_probability(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Return a finite probability in ``[0, 1]`` or ``default``.

    Model outputs can be absent during a rollout or become non-finite after a
    bad feature vector.  The policy must not turn such a value into a trade.
    """
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not isfinite(result):
        return default
    return max(0.0, min(1.0, result))


def _finite_nonnegative(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if isfinite(result) and result >= 0.0 else default


def probability_for_side(p_yes: Optional[float], side: str) -> Optional[float]:
    """Convert a YES probability to the probability of ``side`` winning."""
    p_yes = clamp_probability(p_yes)
    if p_yes is None:
        return None
    if side == BUY_YES:
        return p_yes
    if side == BUY_NO:
        return 1.0 - p_yes
    raise ValueError(f"unsupported side: {side}")


def market_yes_probability(
    *,
    yes_ask: Optional[float],
    no_ask: Optional[float],
    fallback_yes: Optional[float] = None,
) -> Optional[float]:
    """Build a normalized YES prior from both executable sides of the book.

    YES and NO asks normally sum to more than one because both include the
    spread. Normalizing the pair removes that overround and is more stable
    than treating a single side as the market probability.
    """
    yes = clamp_probability(yes_ask)
    no = clamp_probability(no_ask)
    if (
        yes is not None
        and no is not None
        and 0.0 < yes < 1.0
        and 0.0 < no < 1.0
        and yes + no > 0.0
    ):
        return round(yes / (yes + no), 8)
    return clamp_probability(fallback_yes)


def logit(probability: float, epsilon: float = 1e-6) -> float:
    p = clamp_probability(probability, 0.5)
    assert p is not None
    p = max(epsilon, min(1.0 - epsilon, p))
    return log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    # Avoid overflow for a corrupted or extreme runtime value.
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + exp(-value))


_STACKER_FEATURE_NAMES: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class WeightedPolicyConfig:
    """Runtime knobs for the weighted policy.

    The configured 90/5/5 weights are applied in log-odds space. This is
    equivalent to using the market as the prior plus bounded residual
    corrections from LogReg and LightGBM.
    """

    market_weight: float = 0.90
    logreg_weight: float = 0.05
    lgbm_weight: float = 0.05
    mrf_beta: float = 0.0
    models_agree_beta: float = 0.0
    mrf_application: str = "PROBABILITY"
    mrf_sizing_gamma: float = 0.0
    intercept: float = 0.0
    fee_rate: float = 0.07
    maker_fee_rate: float = 0.0
    fee_exponent: float = 1.0
    slippage_rate: float = 0.005
    latency_buffer: float = 0.0
    execution_role: str = "TAKER"
    policy_id: str = "UNVERSIONED"
    mrf_extreme_veto_threshold: float = -1.0
    # Optional immutable ridge-stacker payload loaded from a policy artifact.
    # Empty tuples preserve the initial 90/5/5 residual scorer.
    stacker_feature_names: tuple[str, ...] = ()
    stacker_coefficients: tuple[float, ...] = ()

    def normalized_weights(self, available: set[str]) -> dict[str, float]:
        """Keep the market as prior and absorb missing model weight into it."""
        configured = {
            "market": _finite_nonnegative(self.market_weight),
            "logreg": _finite_nonnegative(self.logreg_weight),
            "lgbm": _finite_nonnegative(self.lgbm_weight),
        }
        total = sum(configured.values())
        if total <= 0.0 or "market" not in available:
            return {name: 0.0 for name in configured}
        normalized = {name: value / total for name, value in configured.items()}
        logreg_weight = normalized["logreg"] if "logreg" in available else 0.0
        lgbm_weight = normalized["lgbm"] if "lgbm" in available else 0.0
        return {
            "market": max(0.0, 1.0 - logreg_weight - lgbm_weight),
            "logreg": logreg_weight,
            "lgbm": lgbm_weight,
        }


@dataclass(frozen=True)
class ProbabilityInputs:
    """Canonical, already-extracted scorer inputs for one market snapshot."""

    p_market_yes: Optional[float]
    p_logreg_yes: Optional[float]
    p_lgbm_yes: Optional[float]
    ece: float = 0.0
    asset: Optional[str] = None
    phase: Optional[str] = None
    regime: Optional[str] = None
    role: Optional[str] = None
    time_left_sec: Optional[float] = None

    @property
    def models_agree(self) -> Optional[bool]:
        p_lr = clamp_probability(self.p_logreg_yes)
        p_lgbm = clamp_probability(self.p_lgbm_yes)
        if p_lr is None or p_lgbm is None:
            return None
        return (p_lr >= 0.5) == (p_lgbm >= 0.5)


@dataclass(frozen=True)
class TradeCostEstimate:
    """Estimated cost of buying one share at ``price``."""

    price: float
    role: str
    fee_rate: float
    maker_fee_rate: float
    fee_exponent: float
    fee_per_share: float
    maker_fee_per_share: float
    taker_fee_per_share: float
    slippage_per_share: float
    spread_per_share: float
    latency_buffer_per_share: float
    total_per_share: float
    expected_execution_price: float
    source: str = "CONFIG_DEFAULT"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def polymarket_taker_fee_per_share(
    price: float, fee_rate: float, fee_exponent: float = 1.0
) -> float:
    """Estimate the CLOB taker fee for one share.

    Polymarket's crypto fee schedule is price-dependent.  The configured rate
    is multiplied by ``(p * (1-p)) ** e``.  ``e`` is supplied by the CLOB
    market's fee curve metadata and defaults to one for compatibility.  This
    remains an estimate; the execution gateway's reported fill fee is
    authoritative.
    """
    p = clamp_probability(price, 0.5)
    rate = _finite_nonnegative(fee_rate)
    exponent = _finite_nonnegative(fee_exponent, 1.0)
    assert p is not None
    return rate * (p * (1.0 - p)) ** exponent


def estimate_trade_cost(
    price: float,
    *,
    fee_rate: float = 0.07,
    maker_fee_rate: float = 0.0,
    fee_exponent: float = 1.0,
    slippage_rate: float = 0.005,
    role: str = "TAKER",
    spread: float = 0.0,
    latency_buffer: float = 0.0,
    source: str = "CONFIG_DEFAULT",
) -> TradeCostEstimate:
    """Build a conservative per-share cost estimate.

    The quote is already a best ask in the decision path, so spread is zero by
    default.  Callers working from a mid-price can pass half-spread (or their
    own calibrated adverse-selection buffer) explicitly.
    """
    price = clamp_probability(price, 0.5)
    assert price is not None
    role = str(role or "TAKER").strip().upper()
    if role not in {"MAKER", "TAKER"}:
        role = "TAKER"
    fee_rate = _finite_nonnegative(fee_rate)
    maker_fee_rate = _finite_nonnegative(maker_fee_rate)
    fee_exponent = _finite_nonnegative(fee_exponent, 1.0)
    slippage_rate = _finite_nonnegative(slippage_rate)
    spread_per_share = _finite_nonnegative(spread)
    latency_per_share = _finite_nonnegative(latency_buffer)
    taker_fee = polymarket_taker_fee_per_share(
        price, fee_rate, fee_exponent
    )
    maker_fee = polymarket_taker_fee_per_share(
        price, maker_fee_rate, fee_exponent
    )
    fee = maker_fee if role == "MAKER" else taker_fee
    slippage = price * slippage_rate
    total = fee + slippage + spread_per_share + latency_per_share
    return TradeCostEstimate(
        price=price,
        role=role,
        fee_rate=fee_rate,
        maker_fee_rate=maker_fee_rate,
        fee_exponent=fee_exponent,
        fee_per_share=round(fee, 8),
        maker_fee_per_share=round(maker_fee, 8),
        taker_fee_per_share=round(taker_fee, 8),
        slippage_per_share=round(slippage, 8),
        spread_per_share=round(spread_per_share, 8),
        latency_buffer_per_share=round(latency_per_share, 8),
        total_per_share=round(total, 8),
        expected_execution_price=round(price + slippage + spread_per_share + latency_per_share, 8),
        source=source,
    )


def compute_net_ev_per_share(
    p_win: float,
    ask: float,
    costs: TradeCostEstimate,
) -> float:
    """Return expected USDC profit for one binary share."""
    probability = clamp_probability(p_win)
    price = clamp_probability(ask)
    if probability is None or price is None:
        raise ValueError("p_win and ask must be finite probabilities")
    return round(probability - price - costs.total_per_share, 8)


def _stacker_score(
    inputs: ProbabilityInputs,
    config: WeightedPolicyConfig,
    signed_evidence: float,
) -> Optional[tuple[float, float, float, float, float, float]]:
    """Evaluate a validated artifact stacker and return auditable terms.

    The feature order is shared with fit_ridge_logistic_stacker. A
    malformed/legacy artifact falls back to the bounded residual scorer rather
    than changing runtime behaviour.
    """
    names = tuple(config.stacker_feature_names or ())
    coefficients_raw = tuple(config.stacker_coefficients or ())
    if names != _STACKER_FEATURE_NAMES or len(coefficients_raw) != len(names):
        return None
    try:
        coefficients = tuple(float(value) for value in coefficients_raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(isfinite(value) for value in coefficients):
        return None
    market = clamp_probability(inputs.p_market_yes)
    if market is None:
        return None
    market_logit = logit(market)
    logreg = clamp_probability(inputs.p_logreg_yes)
    lgbm = clamp_probability(inputs.p_lgbm_yes)
    agreement = inputs.models_agree
    role = str(inputs.role or "").strip().upper()
    role_outsider = 1.0 if role in {"OUTSIDER", "OUTS", "UNDERDOG"} else 0.0
    models_agree = 1.0 if agreement is True else 0.0
    logreg_residual = (
        logit(logreg) - market_logit if logreg is not None else 0.0
    )
    lgbm_residual = (
        logit(lgbm) - market_logit if lgbm is not None else 0.0
    )
    features = (
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
    terms = tuple(
        coefficient * feature
        for coefficient, feature in zip(coefficients, features)
    )
    score = sum(terms)
    # Keep the existing telemetry schema while exposing all terms: role and
    # role×residual terms are grouped with the intercept contribution, and the
    # agreement interaction is grouped with the models-agree contribution.
    return (
        sigmoid(score),
        terms[1],
        terms[2],
        terms[3],
        terms[4],
        terms[0] + terms[5] + terms[8] + terms[9],
    )


@dataclass(frozen=True)
class WeightedProbability:
    p_market_yes: Optional[float]
    p_logreg_yes: Optional[float]
    p_lgbm_yes: Optional[float]
    p_final_yes: float
    market_weight: float
    logreg_weight: float
    lgbm_weight: float
    mrf_evidence: float
    mrf_adjustment_logodds: float
    models_agree_adjustment_logodds: float
    market_contribution_logodds: float
    logreg_contribution_logodds: float
    lgbm_contribution_logodds: float
    intercept_contribution_logodds: float
    models_agree: Optional[bool]
    missing_components: tuple[str, ...]

    @property
    def contributions(self) -> dict[str, float]:
        """Return additive log-odds contributions for telemetry."""
        return {
            "market": self.market_contribution_logodds,
            "logreg": self.logreg_contribution_logodds,
            "lgbm": self.lgbm_contribution_logodds,
            "mrf": self.mrf_adjustment_logodds,
            "models_agree": self.models_agree_adjustment_logodds,
            "intercept": self.intercept_contribution_logodds,
        }

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["missing_components"] = list(self.missing_components)
        result["contributions"] = self.contributions
        return result


def score_weighted_probability(
    *,
    p_market_yes: Optional[float],
    p_logreg_yes: Optional[float],
    p_lgbm_yes: Optional[float],
    config: WeightedPolicyConfig,
    mrf_evidence: Optional[float] = None,
    asset: Optional[str] = None,
    phase: Optional[str] = None,
    regime: Optional[str] = None,
    role: Optional[str] = None,
    time_left_sec: Optional[float] = None,
) -> WeightedProbability:
    """Apply the regularized market-prior residual formula in log-odds."""
    inputs = ProbabilityInputs(
        p_market_yes=p_market_yes,
        p_logreg_yes=p_logreg_yes,
        p_lgbm_yes=p_lgbm_yes,
        asset=asset,
        phase=phase,
        regime=regime,
        role=role,
        time_left_sec=time_left_sec,
    )
    market = clamp_probability(inputs.p_market_yes)
    logreg = clamp_probability(inputs.p_logreg_yes)
    lgbm = clamp_probability(inputs.p_lgbm_yes)
    values = {"market": market, "logreg": logreg, "lgbm": lgbm}
    available = {name for name, value in values.items() if value is not None}
    weights = config.normalized_weights(available)

    try:
        signed_evidence = float(mrf_evidence) if mrf_evidence is not None else 0.0
    except (TypeError, ValueError, OverflowError):
        signed_evidence = 0.0
    if not isfinite(signed_evidence):
        signed_evidence = 0.0
    signed_evidence = max(-1.0, min(1.0, signed_evidence))
    try:
        beta = float(config.mrf_beta)
    except (TypeError, ValueError, OverflowError):
        beta = 0.0
    if not isfinite(beta):
        beta = 0.0
    try:
        intercept = float(config.intercept)
    except (TypeError, ValueError, OverflowError):
        intercept = 0.0
    if not isfinite(intercept):
        intercept = 0.0
    adjustment = beta * signed_evidence
    try:
        agreement_beta = float(config.models_agree_beta)
    except (TypeError, ValueError, OverflowError):
        agreement_beta = 0.0
    if not isfinite(agreement_beta):
        agreement_beta = 0.0
    agreement_adjustment = (
        max(-2.0, min(2.0, agreement_beta))
        if inputs.models_agree is True
        else 0.0
    )

    stacker = _stacker_score(inputs, config, signed_evidence)
    if stacker is not None:
        (
            final,
            market_contribution,
            logreg_contribution,
            lgbm_contribution,
            adjustment,
            intercept,
        ) = stacker
        agreement_adjustment = (
            (
                float(config.stacker_coefficients[6])
                + float(config.stacker_coefficients[7])
            )
            if inputs.models_agree is True
            else 0.0
        )
    elif market is None or sum(weights.values()) <= 0.0:
        market_contribution = 0.0
        logreg_contribution = 0.0
        lgbm_contribution = 0.0
        final = 0.5
    else:
        market_logit = logit(market)
        market_contribution = market_logit
        logreg_contribution = (
            weights["logreg"] * (logit(logreg) - market_logit)
            if logreg is not None and weights["logreg"] > 0.0
            else 0.0
        )
        lgbm_contribution = (
            weights["lgbm"] * (logit(lgbm) - market_logit)
            if lgbm is not None and weights["lgbm"] > 0.0
            else 0.0
        )
        final = sigmoid(
            market_contribution
            + logreg_contribution
            + lgbm_contribution
            + adjustment
            + agreement_adjustment
            + intercept
        )
    missing = tuple(name for name in ("market", "logreg", "lgbm") if name not in available)
    return WeightedProbability(
        p_market_yes=market,
        p_logreg_yes=logreg,
        p_lgbm_yes=lgbm,
        p_final_yes=round(max(0.0, min(1.0, final)), 8),
        market_weight=round(weights["market"], 8),
        logreg_weight=round(weights["logreg"], 8),
        lgbm_weight=round(weights["lgbm"], 8),
        mrf_evidence=round(signed_evidence, 8),
        mrf_adjustment_logodds=round(adjustment, 8),
        models_agree_adjustment_logodds=round(agreement_adjustment, 8),
        market_contribution_logodds=round(market_contribution, 8),
        logreg_contribution_logodds=round(logreg_contribution, 8),
        lgbm_contribution_logodds=round(lgbm_contribution, 8),
        intercept_contribution_logodds=round(intercept, 8),
        models_agree=inputs.models_agree,
        missing_components=missing,
    )


@dataclass(frozen=True)
class WeightedSideQuote:
    side: str
    ask: float
    p_win: float
    gross_ev_per_share: float
    cost: TradeCostEstimate
    net_ev_per_share: float

    @property
    def net_edge(self) -> float:
        """Compatibility alias; edge is always per-share USDC in this policy."""
        return self.net_ev_per_share

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["cost"] = self.cost.as_dict()
        result["net_edge"] = round(self.net_edge, 8)
        return result


@dataclass(frozen=True)
class WeightedSelection:
    probability: WeightedProbability
    selected: Optional[WeightedSideQuote]
    yes_quote: Optional[WeightedSideQuote]
    no_quote: Optional[WeightedSideQuote]
    reason: str

    @property
    def candidate_side(self) -> Optional[str]:
        return self.selected.side if self.selected else None

    @property
    def candidate_ask(self) -> Optional[float]:
        return self.selected.ask if self.selected else None

    @property
    def p_candidate_win(self) -> Optional[float]:
        return self.selected.p_win if self.selected else None

    @property
    def net_ev_per_share(self) -> Optional[float]:
        return self.selected.net_ev_per_share if self.selected else None

    @property
    def best_quote(self) -> Optional[WeightedSideQuote]:
        """Return the best available quote, including below-threshold skips."""
        quotes = [quote for quote in (self.yes_quote, self.no_quote) if quote is not None]
        return max(quotes, key=lambda quote: quote.net_ev_per_share) if quotes else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability.as_dict(),
            "selected": self.selected.as_dict() if self.selected else None,
            "yes_quote": self.yes_quote.as_dict() if self.yes_quote else None,
            "no_quote": self.no_quote.as_dict() if self.no_quote else None,
            "reason": self.reason,
        }


def _make_side_quote(
    side: str,
    ask: Optional[float],
    p_yes: float,
    config: WeightedPolicyConfig,
    *,
    spread: float = 0.0,
    source: str = "CONFIG_DEFAULT",
) -> Optional[WeightedSideQuote]:
    ask = clamp_probability(ask)
    if ask is None or ask <= 0.0 or ask >= 1.0:
        return None
    p_win = probability_for_side(p_yes, side)
    assert p_win is not None
    cost = estimate_trade_cost(
        ask,
        fee_rate=config.fee_rate,
        maker_fee_rate=config.maker_fee_rate,
        fee_exponent=config.fee_exponent,
        slippage_rate=config.slippage_rate,
        role=config.execution_role,
        spread=spread,
        latency_buffer=config.latency_buffer,
        source=source,
    )
    gross_ev = p_win - ask
    net_ev = compute_net_ev_per_share(p_win, ask, cost)
    return WeightedSideQuote(
        side=side,
        ask=round(ask, 8),
        p_win=round(p_win, 8),
        gross_ev_per_share=round(gross_ev, 8),
        cost=cost,
        net_ev_per_share=round(net_ev, 8),
    )


def select_weighted_side(
    *,
    p_market_yes: Optional[float],
    p_logreg_yes: Optional[float],
    p_lgbm_yes: Optional[float],
    yes_ask: Optional[float],
    no_ask: Optional[float],
    config: WeightedPolicyConfig,
    mrf_evidence: Optional[float] = None,
    asset: Optional[str] = None,
    phase: Optional[str] = None,
    regime: Optional[str] = None,
    role: Optional[str] = None,
    time_left_sec: Optional[float] = None,
    min_net_ev: float = 0.0,
    fee_source: str = "CONFIG_DEFAULT",
    spread: float = 0.0,
    mrf_extreme_veto_threshold: Optional[float] = None,
) -> WeightedSelection:
    """Select the side with the highest positive cost-aware expected value.

    ``spread`` is an absolute per-share adverse-selection buffer.  Callers
    using a midpoint prior can pass the observed top-of-book spread; callers
    already using a fully executable ask may leave it at zero.
    """
    probability = score_weighted_probability(
        p_market_yes=p_market_yes,
        p_logreg_yes=p_logreg_yes,
        p_lgbm_yes=p_lgbm_yes,
        config=config,
        mrf_evidence=mrf_evidence,
        asset=asset,
        phase=phase,
        regime=regime,
        role=role,
        time_left_sec=time_left_sec,
    )
    yes_quote = _make_side_quote(
        BUY_YES, yes_ask, probability.p_final_yes, config, spread=spread, source=fee_source,
    )
    no_quote = _make_side_quote(
        BUY_NO, no_ask, probability.p_final_yes, config, spread=spread, source=fee_source,
    )
    quotes = [quote for quote in (yes_quote, no_quote) if quote is not None]
    if not quotes:
        return WeightedSelection(probability, None, yes_quote, no_quote, "NO_VALID_ASK")
    if mrf_extreme_veto_threshold is not None:
        try:
            veto_threshold = float(mrf_extreme_veto_threshold)
        except (TypeError, ValueError, OverflowError):
            veto_threshold = -1.0
        if isfinite(veto_threshold):
            veto_threshold = max(-1.0, min(0.0, veto_threshold))
            # -1.0 is the explicit disabled sentinel; any higher threshold
            # enables a veto for strongly negative MRF evidence.
            if veto_threshold > -1.0 and probability.mrf_evidence <= veto_threshold:
                return WeightedSelection(
                    probability, None, yes_quote, no_quote, "MRF_EXTREME_VETO"
                )
    if (
        probability.market_weight
        + probability.logreg_weight
        + probability.lgbm_weight
        <= 0.0
    ):
        return WeightedSelection(
            probability,
            None,
            yes_quote,
            no_quote,
            "NO_CONFIGURED_COMPONENT_WEIGHT",
        )
    selected = max(quotes, key=lambda quote: quote.net_ev_per_share)
    threshold = _finite_nonnegative(min_net_ev)
    if selected.net_ev_per_share < threshold:
        return WeightedSelection(
            probability,
            None,
            yes_quote,
            no_quote,
            f"BEST_NET_EV_BELOW_THRESHOLD:{selected.net_ev_per_share:.6f}<{threshold:.6f}",
        )
    return WeightedSelection(probability, selected, yes_quote, no_quote, "SELECTED_BEST_NET_EV")


def logreg_flip_to_yes_probability(
    p_flip: Optional[float],
    fresh_yes_price: Optional[float],
) -> Optional[float]:
    """Convert LogReg's P(flip) to P(YES wins) using the current favorite."""
    p_flip = clamp_probability(p_flip)
    market = clamp_probability(fresh_yes_price)
    if p_flip is None or market is None:
        return None
    yes_is_favorite = market >= 0.50
    return round((1.0 - p_flip) if yes_is_favorite else p_flip, 8)



def benchmark_policy_arms(
    *,
    p_market_yes: Optional[float],
    p_logreg_yes: Optional[float],
    p_lgbm_yes: Optional[float],
    yes_ask: Optional[float],
    no_ask: Optional[float],
    config: WeightedPolicyConfig,
    mrf_evidence: Optional[float] = None,
    asset: Optional[str] = None,
    phase: Optional[str] = None,
    regime: Optional[str] = None,
    role: Optional[str] = None,
    time_left_sec: Optional[float] = None,
    fee_source: str = "CONFIG_DEFAULT",
    spread: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Return compact counterfactual results for every rollout arm.

    All arms consume one probability/quote snapshot. The helper is deliberately
    pure so it can be used by runtime shadow telemetry and offline checks.
    """
    common = {
        "fee_rate": config.fee_rate,
        "maker_fee_rate": config.maker_fee_rate,
        "fee_exponent": config.fee_exponent,
        "slippage_rate": config.slippage_rate,
        "latency_buffer": config.latency_buffer,
        "execution_role": config.execution_role,
        "policy_id": config.policy_id,
        "mrf_extreme_veto_threshold": config.mrf_extreme_veto_threshold,
    }
    arms = {
        "MARKET_ONLY": replace(
            config, market_weight=1.0, logreg_weight=0.0, lgbm_weight=0.0,
            mrf_beta=0.0, models_agree_beta=0.0,
            stacker_feature_names=(), stacker_coefficients=(), **common,
        ),
        "MARKET_LOGREG": replace(
            config, market_weight=0.8, logreg_weight=0.2, lgbm_weight=0.0,
            mrf_beta=0.0, models_agree_beta=0.0,
            stacker_feature_names=(), stacker_coefficients=(), **common,
        ),
        "MARKET_LGBM": replace(
            config, market_weight=0.8, logreg_weight=0.0, lgbm_weight=0.2,
            mrf_beta=0.0, models_agree_beta=0.0,
            stacker_feature_names=(), stacker_coefficients=(), **common,
        ),
        "FULL_WEIGHTED": replace(
            config, mrf_beta=0.0, models_agree_beta=0.0, **common,
        ),
        "FULL_WEIGHTED_MRF": replace(config, **common),
    }
    result: dict[str, dict[str, Any]] = {}

    def compact(name: str, selection: WeightedSelection) -> dict[str, Any]:
        selected = selection.selected
        return {
            "arm": name,
            "selected_side": selected.side if selected else None,
            "selected_ask": selected.ask if selected else None,
            "p_final_yes": selection.probability.p_final_yes,
            "models_agree": selection.probability.models_agree,
            "yes_net_ev": selection.yes_quote.net_ev_per_share if selection.yes_quote else None,
            "no_net_ev": selection.no_quote.net_ev_per_share if selection.no_quote else None,
            "net_ev_per_share": selected.net_ev_per_share if selected else None,
            "reason": selection.reason,
            "missing_components": list(selection.probability.missing_components),
        }

    for name, arm_config in arms.items():
        selection = select_weighted_side(
            p_market_yes=p_market_yes,
            p_logreg_yes=p_logreg_yes,
            p_lgbm_yes=p_lgbm_yes,
            yes_ask=yes_ask,
            no_ask=no_ask,
            config=arm_config,
            mrf_evidence=mrf_evidence if name == "FULL_WEIGHTED_MRF" else None,
            asset=asset,
            phase=phase,
            regime=regime,
            role=role,
            time_left_sec=time_left_sec,
            min_net_ev=0.0,
            fee_source=fee_source,
            spread=spread,
            mrf_extreme_veto_threshold=arm_config.mrf_extreme_veto_threshold,
        )
        result[name] = compact(name, selection)

    agreement = (
        p_logreg_yes is not None
        and p_lgbm_yes is not None
        and (p_logreg_yes >= 0.5) == (p_lgbm_yes >= 0.5)
    )
    outsider_selection = result.get("FULL_WEIGHTED_MRF")
    if not agreement:
        result["OUTSIDER_AGREE_ONLY"] = {
            "arm": "OUTSIDER_AGREE_ONLY",
            "selected_side": None,
            "p_final_yes": outsider_selection.get("p_final_yes") if outsider_selection else None,
            "models_agree": False if p_logreg_yes is not None and p_lgbm_yes is not None else None,
            "yes_net_ev": outsider_selection.get("yes_net_ev") if outsider_selection else None,
            "no_net_ev": outsider_selection.get("no_net_ev") if outsider_selection else None,
            "net_ev_per_share": None,
            "reason": "OUTSIDER_AGREE_GATE",
            "missing_components": [],
        }
    elif outsider_selection is None or outsider_selection.get("selected_side") is None:
        result["OUTSIDER_AGREE_ONLY"] = {
            "arm": "OUTSIDER_AGREE_ONLY",
            "selected_side": None,
            "p_final_yes": outsider_selection.get("p_final_yes") if outsider_selection else None,
            "models_agree": True,
            "yes_net_ev": outsider_selection.get("yes_net_ev") if outsider_selection else None,
            "no_net_ev": outsider_selection.get("no_net_ev") if outsider_selection else None,
            "net_ev_per_share": None,
            "reason": outsider_selection.get("reason", "NO_SELECTION") if outsider_selection else "NO_SELECTION",
            "missing_components": [],
        }
    else:
        selected_ask = float(outsider_selection.get("selected_ask") or 0.0)
        is_outsider = selected_ask < 0.50
        result["OUTSIDER_AGREE_ONLY"] = (
            dict(outsider_selection, arm="OUTSIDER_AGREE_ONLY")
            if is_outsider
            else dict(outsider_selection, arm="OUTSIDER_AGREE_ONLY", selected_side=None, net_ev_per_share=None, reason="NOT_OUTSIDER")
        )
    return result
