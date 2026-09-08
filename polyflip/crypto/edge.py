"""
Модуль вычисления силы сигнала для крипто-модели LightGBM и экономического преимущества.
"""
from __future__ import annotations
import numpy as np

def compute_crypto_signal_strength(
    p_up: float,
    threshold_up: float,
    threshold_down: float,
) -> tuple[float, str]:
    """
    Вычисляет силу крипто-сигнала с учетом асимметричных порогов.

    Returns:
        (signal_strength, direction) где direction: "UP" | "DOWN" | "NONE".
        Если p_up попадает в мертвую зону (threshold_down, threshold_up),
        возвращается signal_strength=0.0 и direction="NONE".
    """
    if not 0.0 <= threshold_down < threshold_up <= 1.0:
        return 0.0, "NONE"
    # ``threshold_down`` is a lower bound on p_up, not a threshold on p_down.
    # Keeping both thresholds in the same coordinate system makes the dead
    # zone explicit and prevents overlapping UP/DOWN ranges.
    if p_up >= threshold_up:
        return round(p_up - threshold_up, 4), "UP"
    if p_up <= threshold_down:
        return round(threshold_down - p_up, 4), "DOWN"
    return 0.0, "NONE"

def compute_net_ev_per_share(
    p_win: float,
    executable_ask: float,
    fee_per_share: float = 0.0,
    slippage_per_share: float = 0.0,
    latency_buffer_per_share: float = 0.0,
    spread_buffer_per_share: float = 0.0,
) -> float:
    """
    Unified canonical EV calculation:
    net_EV_per_share = p_win - executable_ask - entry_fee_per_share - additional_costs.

    Note: executable_ask already reflects buying at the ask (paying half the spread);
    full spread must NOT be subtracted a second time.
    """
    p = float(np.clip(p_win, 0.0, 1.0))
    ask = float(executable_ask)
    if ask <= 0.0 or ask >= 1.0:
        return 0.0
    costs = float(fee_per_share + slippage_per_share + latency_buffer_per_share + spread_buffer_per_share)
    net_ev = p - ask - costs
    return round(float(net_ev), 8)


def compute_economic_edge(
    p_win: float,
    executable_ask: float,
    fee_rate: float,
    slippage_rate: float,
    latency_buffer: float = 0.0,
) -> float:
    """
    Calculates net EV per share using the unified formula (USDC per share).
    Replaces legacy ROI formula gross_edge = p_win / cost - 1 with true net EV.
    """
    if executable_ask <= 0.0 or executable_ask >= 1.0:
        return 0.0
    fee_per_share = executable_ask * fee_rate
    slippage_per_share = executable_ask * slippage_rate
    return compute_net_ev_per_share(
        p_win=p_win,
        executable_ask=executable_ask,
        fee_per_share=fee_per_share,
        slippage_per_share=slippage_per_share,
        latency_buffer_per_share=latency_buffer,
    )


def compute_roi(
    net_ev_per_share: float,
    executable_ask: float,
) -> float:
    """Expected return on capital: net_ev_per_share / executable_ask."""
    if executable_ask <= 0.0:
        return 0.0
    return float(net_ev_per_share / executable_ask)

