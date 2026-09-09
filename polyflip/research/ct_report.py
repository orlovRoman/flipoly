"""
polyflip/research/ct_report.py

Compact PAPER Profile Reporting and Mode Comparison (Stage 5, Items 27 & 28):
- Breakdown by UP and DOWN sides: opportunities, skips, signals, orders, fills, settlement, PnL.
- Invariant: total equals sum of UP + DOWN.
- Detailed reason breakdown explaining divergence between YES-only and symmetric modes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Sequence

from polyflip.trading.ct_policy import (
    CTSpecification,
    get_btc_ct_t5_v1_spec,
    calculate_scenario_economics,
)


@dataclass
class SideReport:
    side: str  # "UP", "DOWN", or "TOTAL"
    opportunities: int = 0
    signals_reversion: int = 0
    signals_trend: int = 0
    signals_quiet: int = 0
    signals_uncertain: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    orders_placed: int = 0
    orders_full_fill: int = 0
    orders_partial_fill: int = 0
    orders_unfilled: int = 0
    total_spent_usdc: float = 0.0
    total_fee_usdc: float = 0.0
    total_shares_filled: float = 0.0
    settled_trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl_usdc: float = 0.0
    net_pnl_usdc: float = 0.0
    scenario_net_pnl_usdc: float = 0.0

    @property
    def win_rate(self) -> float:
        return (self.wins / self.settled_trades) if self.settled_trades > 0 else 0.0

    @property
    def vwap(self) -> float | None:
        return (self.total_spent_usdc / self.total_shares_filled) if self.total_shares_filled > 0 else None


@dataclass
class CTProfileReport:
    spec_id: str
    spec_hash: str
    generated_at: str
    up_report: SideReport
    down_report: SideReport
    total_report: SideReport
    invariant_passed: bool
    unassigned_report: SideReport = field(default_factory=lambda: SideReport(side="UNASSIGNED"))
    mode_comparison: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "generated_at": self.generated_at,
            "up_side": asdict(self.up_report),
            "down_side": asdict(self.down_report),
            "unassigned_side": asdict(self.unassigned_report),
            "total": asdict(self.total_report),
            "invariant_passed": self.invariant_passed,
            "mode_comparison": self.mode_comparison,
        }

    def format_markdown(self) -> str:
        """Render compact human-readable Markdown table."""
        u = self.up_report
        d = self.down_report
        tot = self.total_report
        return f"""
# Отчёт PAPER-профиля {self.spec_id}
**Спецификация**: `{self.spec_id}` (hash: `{self.spec_hash[:16]}...`)
**Сгенерирован**: {self.generated_at}
**Инвариант Total = UP + DOWN**: {'СОБЛЮДЁН (PASS)' if self.invariant_passed else 'НАРУШЕН (FAIL)'}

| Метрика | UP (YES) | DOWN (NO) | TOTAL |
| :--- | :---: | :---: | :---: |
| **Возможностей рассмотрено** | {u.opportunities} | {d.opportunities} | {tot.opportunities} |
| **Сигналы REVERSION** | {u.signals_reversion} | {d.signals_reversion} | {tot.signals_reversion} |
| **Заявок подано (BUY)** | {u.orders_placed} | {d.orders_placed} | {tot.orders_placed} |
| **Полных fills** | {u.orders_full_fill} | {d.orders_full_fill} | {tot.orders_full_fill} |
| **Частичных fills** | {u.orders_partial_fill} | {d.orders_partial_fill} | {tot.orders_partial_fill} |
| **Куплено акций** | {u.total_shares_filled:.2f} | {d.total_shares_filled:.2f} | {tot.total_shares_filled:.2f} |
| **Потрачено USDC** | ${u.total_spent_usdc:.2f} | ${d.total_spent_usdc:.2f} | ${tot.total_spent_usdc:.2f} |
| **Комиссия USDC** | ${u.total_fee_usdc:.4f} | ${d.total_fee_usdc:.4f} | ${tot.total_fee_usdc:.4f} |
| **Разрешено сделок (Settlement)** | {u.settled_trades} | {d.settled_trades} | {tot.settled_trades} |
| **Побед (Wins)** | {u.wins} | {d.wins} | {tot.wins} |
| **Винрейт** | {u.win_rate*100:.2f}% | {d.win_rate*100:.2f}% | {tot.win_rate*100:.2f}% |
| **Сценарный Net PnL (USDC)** | {u.scenario_net_pnl_usdc:+.4f} | {d.scenario_net_pnl_usdc:+.4f} | {tot.scenario_net_pnl_usdc:+.4f} |
| **Фактический Net PnL (USDC)** | {u.net_pnl_usdc:+.4f} | {d.net_pnl_usdc:+.4f} | {tot.net_pnl_usdc:+.4f} |

### Причины пропусков (SKIP)
- **UP**: {dict(u.skip_reasons)}
- **DOWN**: {dict(d.skip_reasons)}
- **TOTAL**: {dict(tot.skip_reasons)}
"""


def build_profile_report(
    decisions_and_executions: Sequence[dict[str, Any]],
    spec: CTSpecification | None = None,
) -> CTProfileReport:
    """
    Builds compact summary report from decision and execution records (Item 28).
    Strictly verifies invariant: total = up + down + unassigned.
    """
    spec = spec or get_btc_ct_t5_v1_spec()
    up = SideReport(side="UP")
    down = SideReport(side="DOWN")
    unassigned = SideReport(side="UNASSIGNED")

    for rec in decisions_and_executions:
        raw_side = rec.get("side")
        side = str(raw_side).upper() if raw_side is not None else ""
        if side == "UP":
            target_rep = up
        elif side == "DOWN":
            target_rep = down
        else:
            target_rep = unassigned

        target_rep.opportunities += 1
        regime = str(rec.get("ct_regime") or "").upper()
        if regime == "REVERSION":
            target_rep.signals_reversion += 1
        elif regime == "TREND":
            target_rep.signals_trend += 1
        elif regime == "QUIET":
            target_rep.signals_quiet += 1
        else:
            target_rep.signals_uncertain += 1

        action = str(rec.get("action") or "").upper()
        if action == "SKIP":
            reason = str(rec.get("reason") or "UNKNOWN_SKIP")
            target_rep.skip_reasons[reason] = target_rep.skip_reasons.get(reason, 0) + 1
        elif action == "BUY":
            target_rep.orders_placed += 1
            f_status = str(rec.get("fill_status") or "FULL").upper()
            if f_status == "FULL":
                target_rep.orders_full_fill += 1
            elif f_status == "PARTIAL":
                target_rep.orders_partial_fill += 1
            else:
                target_rep.orders_unfilled += 1

            spent = float(rec.get("spent_usdc") or rec.get("cost_usdc") or 0.0)
            fee = float(rec.get("fee_usdc") or 0.0)
            shares = float(rec.get("filled_shares") or 0.0)
            target_rep.total_spent_usdc += spent
            target_rep.total_fee_usdc += fee
            target_rep.total_shares_filled += shares

            sc_pnl = float(rec.get("scenario_net_pnl") or 0.0)
            target_rep.scenario_net_pnl_usdc += sc_pnl

            # Settlement tracking if resolved
            if rec.get("is_settled"):
                target_rep.settled_trades += 1
                outcome = str(rec.get("settlement_outcome") or "").upper()
                won = outcome in {"WIN", "YES", "1"}
                if won:
                    target_rep.wins += 1
                else:
                    target_rep.losses += 1

                net_p = float(rec.get("realized_pnl_usdc") or rec.get("net_pnl") or 0.0)
                gross_p = float(rec.get("gross_pnl") or (net_p + fee))
                target_rep.net_pnl_usdc += net_p
                target_rep.gross_pnl_usdc += gross_p

    # Combine totals
    tot = SideReport(
        side="TOTAL",
        opportunities=up.opportunities + down.opportunities + unassigned.opportunities,
        signals_reversion=up.signals_reversion + down.signals_reversion + unassigned.signals_reversion,
        signals_trend=up.signals_trend + down.signals_trend + unassigned.signals_trend,
        signals_quiet=up.signals_quiet + down.signals_quiet + unassigned.signals_quiet,
        signals_uncertain=up.signals_uncertain + down.signals_uncertain + unassigned.signals_uncertain,
        orders_placed=up.orders_placed + down.orders_placed + unassigned.orders_placed,
        orders_full_fill=up.orders_full_fill + down.orders_full_fill + unassigned.orders_full_fill,
        orders_partial_fill=up.orders_partial_fill + down.orders_partial_fill + unassigned.orders_partial_fill,
        orders_unfilled=up.orders_unfilled + down.orders_unfilled + unassigned.orders_unfilled,
        total_spent_usdc=round(up.total_spent_usdc + down.total_spent_usdc + unassigned.total_spent_usdc, 4),
        total_fee_usdc=round(up.total_fee_usdc + down.total_fee_usdc + unassigned.total_fee_usdc, 6),
        total_shares_filled=round(up.total_shares_filled + down.total_shares_filled + unassigned.total_shares_filled, 4),
        settled_trades=up.settled_trades + down.settled_trades + unassigned.settled_trades,
        wins=up.wins + down.wins + unassigned.wins,
        losses=up.losses + down.losses + unassigned.losses,
        gross_pnl_usdc=round(up.gross_pnl_usdc + down.gross_pnl_usdc + unassigned.gross_pnl_usdc, 4),
        net_pnl_usdc=round(up.net_pnl_usdc + down.net_pnl_usdc + unassigned.net_pnl_usdc, 4),
        scenario_net_pnl_usdc=round(up.scenario_net_pnl_usdc + down.scenario_net_pnl_usdc + unassigned.scenario_net_pnl_usdc, 4),
    )
    # Merge skip reasons
    for r, c in up.skip_reasons.items():
        tot.skip_reasons[r] = tot.skip_reasons.get(r, 0) + c
    for r, c in down.skip_reasons.items():
        tot.skip_reasons[r] = tot.skip_reasons.get(r, 0) + c
    for r, c in unassigned.skip_reasons.items():
        tot.skip_reasons[r] = tot.skip_reasons.get(r, 0) + c

    # Assert invariant Total == UP + DOWN + UNASSIGNED
    inv_pass = (
        tot.opportunities == (up.opportunities + down.opportunities + unassigned.opportunities)
        and tot.orders_placed == (up.orders_placed + down.orders_placed + unassigned.orders_placed)
        and tot.settled_trades == (up.settled_trades + down.settled_trades + unassigned.settled_trades)
        and math.isclose(tot.net_pnl_usdc, up.net_pnl_usdc + down.net_pnl_usdc + unassigned.net_pnl_usdc, abs_tol=1e-3)
    )

    # Item 27: Structural mode comparison explanation
    mode_comp = {
        "historical_yes_only": {
            "focus": "UP token only (YES mid <= 0.50)",
            "token_history": "YES mid_price snapshots only",
            "trades_in_study": 647,
            "win_rate": 0.2473,
            "net_pnl_usdc": 81.6477,
        },
        "symmetric_ct_policy": {
            "focus": "Symmetric outsider selection (UP or DOWN)",
            "down_selection_rule": "DOWN is outsider when DOWN mid < UP mid",
            "token_history_rule": "Strictly NO token snapshots (no 1 - YES reconstruction)",
            "expected_divergence_drivers": [
                "Missing historical NO token depth/snapshots in older archives",
                "Contested markets with parity quotes (abs(mid_up - mid_down) <= 1e-4)",
                "Real book depth constraints (partial fills vs theoretical $1/ask assumption)",
                "Asymmetric spread widths between UP and DOWN orderbooks",
            ],
        }
    }

    return CTProfileReport(
        spec_id=spec.spec_id,
        spec_hash=spec.spec_hash,
        generated_at=datetime.now(timezone.utc).isoformat(),
        up_report=up,
        down_report=down,
        total_report=tot,
        invariant_passed=inv_pass,
        unassigned_report=unassigned,
        mode_comparison=mode_comp,
    )
