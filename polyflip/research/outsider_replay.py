"""
polyflip/research/outsider_replay.py

Stateful research replay engine with position state machine, ledger accounting,
and single-entry per market policy (Items 1.22, 1.23, R8).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Any, Optional
import numpy as np
import pandas as pd

from polyflip.crypto.edge import compute_net_ev_per_share


@dataclass(frozen=True)
class ReplayPolicy:
    max_positions_per_market: int = 1
    stake_usdc: float = 1.0
    initial_capital: float = 100.0
    min_edge: float = 0.02
    max_price: float = 0.95
    min_price: float = 0.01
    default_fee_rate: float = 0.002
    allow_reentry: bool = False
    veto_mode: bool = False


@dataclass
class ReplayTrade:
    market_id: str
    decision_at: pd.Timestamp
    time_left_min: float
    candidate_side: str
    quote_ask: float
    fill_price: float
    size_shares: float
    spent_usdc: float
    entry_fee: float
    settlement_payout: float
    realized_pnl: float
    outcome: int
    net_ev: float
    execution_role: str = "TAKER"
    provenance: str = "HISTORICAL_REPLAY"


@dataclass
class ReplayLedger:
    executed_trades: list[dict[str, Any]] = field(default_factory=list)
    skipped_decisions: list[dict[str, Any]] = field(default_factory=list)
    vetoed_decisions: list[dict[str, Any]] = field(default_factory=list)
    saved_losses_pnl: float = 0.0
    missed_gains_pnl: float = 0.0
    initial_capital: float = 100.0
    cash_remaining: float = 100.0
    total_pnl: float = 0.0
    total_invested: float = 0.0
    equity_curve: list[float] = field(default_factory=list)

    @property
    def trade_pnls(self) -> list[float]:
        return [t["realized_pnl"] for t in self.executed_trades]

    @property
    def vetoed_trades(self) -> list[dict[str, Any]]:
        return self.vetoed_decisions


class OutsiderReplayEngine:
    """
    Simulates trading policy with position tracking, fee deduction, and capital ledger.
    Guarantees no duplicate entries per market when max_positions_per_market=1.
    """

    def __init__(self, policy: ReplayPolicy | None = None):
        self.policy = policy or ReplayPolicy()


    def run(
        self,
        decision_rows: Sequence[dict[str, Any]] | pd.DataFrame,
        fee_schedule: Any = None,
    ) -> ReplayLedger:
        if isinstance(decision_rows, pd.DataFrame):
            df = decision_rows.copy()
            if "decision_at" in df.columns:
                df["decision_at"] = pd.to_datetime(df["decision_at"], utc=True)
                df = df.sort_values("decision_at")
            records = df.to_dict("records")
        else:
            records = sorted(
                decision_rows,
                key=lambda r: pd.to_datetime(r.get("decision_at", 0), utc=True),
            )

        ledger = ReplayLedger(
            initial_capital=self.policy.initial_capital,
            cash_remaining=self.policy.initial_capital,
        )

        open_markets: dict[str, str] = {}  # market_id -> entered_side
        market_entry_counts: dict[str, int] = {}
        cum_pnl = 0.0
        saved_losses = 0.0
        missed_gains = 0.0
        
        pending_settlements = []

        def process_settlements(current_time):
            nonlocal cum_pnl
            resolved = []
            for settle in pending_settlements:
                if current_time >= settle["settlement_at"]:
                    payout = settle["payout"]
                    stake = settle["stake"]
                    entry_fee = settle["entry_fee"]
                    trade_pnl = payout - stake - entry_fee
                    
                    cum_pnl += trade_pnl
                    ledger.cash_remaining += payout
                    ledger.total_pnl = round(cum_pnl, 4)
                    ledger.equity_curve.append(round(ledger.initial_capital + cum_pnl, 4))
                    resolved.append(settle)
            for r in resolved:
                pending_settlements.remove(r)

        for row in records:
            m_id = str(row.get("market_id", ""))
            side = str(row.get("candidate_side", "UP")).upper()
            tl = float(row.get("time_left_min", 0.0))
            dec_at = pd.to_datetime(row.get("decision_at", pd.Timestamp.now(tz="UTC")), utc=True)
            
            # Process settlements before new decision
            process_settlements(dec_at)

            current_entries = market_entry_counts.get(m_id, 0)
            if not self.policy.allow_reentry and current_entries >= self.policy.max_positions_per_market:
                ledger.skipped_decisions.append({
                    "market_id": m_id,
                    "decision_at": dec_at,
                    "reason": "POSITION_ALREADY_OPEN",
                    "time_left_min": tl,
                })
                continue

            ask = row.get("executable_ask")
            if ask is None or pd.isna(ask):
                ask = row.get("poly_down_best_ask") if side in ("DOWN", "NO") else row.get("poly_up_best_ask")
            
            if ask is None or pd.isna(ask) or ask <= 0.0 or ask >= 1.0:
                ledger.skipped_decisions.append({
                    "market_id": m_id,
                    "decision_at": dec_at,
                    "reason": "MISSING_CANDIDATE_QUOTE",
                    "time_left_min": tl,
                })
                continue

            ask = float(ask)
            if ask > self.policy.max_price or ask < self.policy.min_price:
                ledger.skipped_decisions.append({
                    "market_id": m_id,
                    "decision_at": dec_at,
                    "reason": "PRICE_OUT_OF_BOUNDS",
                    "time_left_min": tl,
                })
                continue

            p_win = row.get("p_win", row.get("calibrated_p", row.get("p_candidate_win")))
            if p_win is None or pd.isna(p_win):
                ledger.skipped_decisions.append({
                    "market_id": m_id,
                    "decision_at": dec_at,
                    "reason": "MISSING_PREDICTION",
                    "time_left_min": tl,
                })
                continue
            p_win = float(p_win)

            if fee_schedule is not None:
                fee_info = fee_schedule.get_fee(
                    market_id=m_id, 
                    decision_at=dec_at, 
                    size_usd=self.policy.stake_usdc, 
                    execution_role="TAKER"
                )
                fee_per_share = fee_info.get("fee_usd", 0.0) / (self.policy.stake_usdc / ask) if self.policy.stake_usdc > 0 else 0.0
                fee_rate = fee_info.get("fee_rate", self.policy.default_fee_rate)
                fee_source = fee_info.get("source", "EXPLICIT_SCHEDULE")
            else:
                fee_rate = float(row.get("fee_rate", self.policy.default_fee_rate))
                fee_per_share = ask * fee_rate
                fee_source = "DEFAULT_ASSUMPTION"

            net_ev = compute_net_ev_per_share(p_win=p_win, ask_price=ask, fee_per_share=fee_per_share)
            if pd.isna(net_ev) or net_ev < self.policy.min_edge:
                ledger.skipped_decisions.append({
                    "market_id": m_id,
                    "decision_at": dec_at,
                    "reason": "INSUFFICIENT_EDGE",
                    "net_ev": net_ev,
                    "time_left_min": tl,
                })
                continue

            # Check veto policy (Item 1.29)
            if self.policy.veto_mode or row.get("is_vetoed"):
                l_dir = str(row.get("lgbm_direction", "NONE")).upper()
                is_vetoed = False
                if side in ("UP", "YES") and l_dir == "DOWN":
                    is_vetoed = True
                elif side in ("DOWN", "NO") and l_dir == "UP":
                    is_vetoed = True
                if row.get("is_vetoed") is True:
                    is_vetoed = True

                if is_vetoed:
                    # Calculate counterfactual result for vetoed entry
                    target_val = row.get("target", row.get("y_candidate_win"))
                    cf_outcome = None
                    if target_val is not None and not pd.isna(target_val):
                        cf_outcome = int(target_val)
                    elif row.get("final_outcome") in ("UP", "YES"):
                        cf_outcome = 1 if side in ("UP", "YES") else 0
                    elif row.get("final_outcome") in ("DOWN", "NO"):
                        cf_outcome = 1 if side in ("DOWN", "NO") else 0

                    cf_pnl = 0.0
                    if cf_outcome is not None:
                        cf_shares = self.policy.stake_usdc / ask
                        cf_fee = self.policy.stake_usdc * fee_rate
                        cf_pnl = (cf_shares * 1.0 - self.policy.stake_usdc - cf_fee) if cf_outcome == 1 else (-self.policy.stake_usdc - cf_fee)
                        if cf_pnl < 0:
                            saved_losses += abs(cf_pnl)
                        else:
                            missed_gains += cf_pnl

                    veto_rec = {
                        "market_id": m_id,
                        "decision_at": dec_at,
                        "reason": "VETOED_BY_LGBM",
                        "time_left_min": tl,
                        "candidate_side": side,
                        "quote_ask": ask,
                        "net_ev": round(net_ev, 4),
                        "counterfactual_pnl": round(cf_pnl, 4),
                        "counterfactual_outcome": cf_outcome,
                    }
                    ledger.skipped_decisions.append(veto_rec)
                    ledger.vetoed_decisions.append(veto_rec)
                    # Crucially do NOT increment market_entry_counts[m_id], allowing later decision points to enter
                    continue

            # Capital Constraints Check (Item 14)
            stake = self.policy.stake_usdc
            fill_price = ask
            shares = stake / fill_price
            entry_fee = fee_per_share * shares
            if ledger.cash_remaining < stake + entry_fee:
                ledger.skipped_decisions.append({
                    "market_id": m_id,
                    "decision_at": dec_at,
                    "reason": "INSUFFICIENT_FUNDS",
                    "time_left_min": tl,
                })
                continue

            # Resolve settlement target (Item 1.04, 1.23: never settle unresolved as loss)
            target_val = row.get("target", row.get("y_candidate_win"))
            outcome = None
            if target_val is not None and not pd.isna(target_val):
                try:
                    outcome = int(target_val)
                except (ValueError, TypeError):
                    outcome = None
            if outcome is None:
                final_out = str(row.get("final_outcome", "")).strip().upper()
                if final_out in ("UP", "YES"):
                    outcome = 1 if side in ("UP", "YES") else 0
                elif final_out in ("DOWN", "NO"):
                    outcome = 1 if side in ("DOWN", "NO") else 0

            if outcome is None or outcome not in (0, 1):
                ledger.skipped_decisions.append({
                    "market_id": m_id,
                    "decision_at": dec_at,
                    "reason": "UNRESOLVED_OUTCOME",
                    "time_left_min": tl,
                })
                continue

            payout = shares * 1.0 if outcome == 1 else 0.0

            # Deduct cash now, put payout in pending settlements
            ledger.cash_remaining -= (stake + entry_fee)
            ledger.total_invested += stake
            
            settlement_at = row.get("market_end_at")
            tl_valid = float(tl) if (tl is not None and np.isfinite(tl) and tl > 0) else 15.0
            if settlement_at is None or pd.isna(settlement_at):
                settlement_at = dec_at + pd.Timedelta(minutes=tl_valid)
            else:
                settlement_at = pd.to_datetime(settlement_at, utc=True)
                if settlement_at < dec_at:
                    settlement_at = dec_at + pd.Timedelta(minutes=tl_valid)
                    
            pending_settlements.append({
                "market_id": m_id,
                "settlement_at": settlement_at,
                "payout": payout,
                "stake": stake,
                "entry_fee": entry_fee
            })

            market_entry_counts[m_id] = current_entries + 1
            open_markets[m_id] = side

            trade_record = {
                "market_id": m_id,
                "decision_at": dec_at,
                "time_left_min": tl,
                "candidate_side": side,
                "quote_ask": ask,
                "fill_price": fill_price,
                "size_shares": round(shares, 4),
                "spent_usdc": round(stake, 4),
                "entry_fee": round(entry_fee, 4),
                "fee_source": fee_source,
                "settlement_payout": round(payout, 4),
                "realized_pnl": round(payout - stake - entry_fee, 4),
                "outcome": outcome,
                "net_ev": round(net_ev, 4),
                "p_win": p_win,
                "fill_provenance": "OBSERVED_QUOTE" if pd.notna(row.get("executable_ask")) else "COUNTERFACTUAL_ASK",
            }
            ledger.executed_trades.append(trade_record)
            
        # Process remaining settlements at the end of time
        process_settlements(pd.Timestamp.max.tz_localize("UTC"))

        ledger.saved_losses_pnl = round(saved_losses, 4)
        ledger.missed_gains_pnl = round(missed_gains, 4)
        return ledger
