"""Contract market rules and question parsing for strike-baseline research.

Settlement understanding (must be verified per-contract, see P1-03):
  Polymarket 15m "Up or Down" binary markets: YES pays 1 if underlying at
  window close > strike; strike is the reference price at window open.
  Exact final-averaging (TWAP) rule is NOT fully retrievable from stored data,
  so history uses proxy strike = first observed Binance 1m close of the window.
  Proxy is NEVER labelled canonical (see P1-05).

Question format (observed):
  "Bitcoin Up or Down - September 7, 7:45PM-8:00PM ET"
  end_time_est == window end (UTC). Window length == 15 minutes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

WINDOW_MIN = 15
_DT_RE = re.compile(
    r"(?P<mon>January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"(?P<day>\d{1,2}),\s+"
    r"(?P<start_h>1?\d):(?P<start_m>\d{2})(?P<start_ampm>AM|PM)-"
    r"(?P<end_h>1?\d):(?P<end_m>\d{2})(?P<end_ampm>AM|PM)\s+ET$"
)
_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def _to_24(h: int, ampm: str) -> int:
    if ampm == "AM":
        return 0 if h == 12 else h
    return 12 if h == 12 else h + 12


def parse_question_window(question: str) -> tuple[int, int, int, int] | None:
    """Returns (month, day, start_h24, start_m, end_h24, end_m) wall-ET or None.

    Year is not part of the question; caller supplies it from end_time_est.
    """
    m = _DT_RE.search(question)
    if not m:
        return None
    return (
        _MONTHS[m.group("mon")],
        int(m.group("day")),
        _to_24(int(m.group("start_h")), m.group("start_ampm")),
        int(m.group("start_m")),
        _to_24(int(m.group("end_h")), m.group("end_ampm")),
        int(m.group("end_m")),
    )


def infer_et_offset(question: str, end_time_est: datetime) -> int | None:
    """Infer ET->UTC offset (hours) such that window end equals end_time_est.

    Tries EDT (-4) then EST (-5). Returns offset, or None on mismatch.
    end_time_est is passed UTC-naive or tz-aware (converted to UTC).
    """
    if end_time_est.tzinfo is not None:
        end_time_est = end_time_est.astimezone(timezone.utc).replace(tzinfo=None)
    parsed = parse_question_window(question)
    if parsed is None:
        return None
    mon, day, sh, sm, eh, em = parsed
    for off in (-4, -5):
        # Window end in UTC = wall(ET) - offset
        try:
            wend_utc = datetime(end_time_est.year, mon, day, eh, em) + timedelta(hours=-off)
        except ValueError:  # e.g. Feb 29 in non-leap year (year comes from end_time_est)
            continue
        if wend_utc == end_time_est:
            return off
    return None


@dataclass(frozen=True)
class MarketWindow:
    market_id: str
    asset: str
    question: str
    window_start_utc: datetime
    window_end_utc: datetime
    et_offset_hours: int | None
    rules_ok: bool
    rules_note: str = ""


def build_window(market_id: str, asset: str, question: str, end_time_est: datetime) -> MarketWindow:
    """Reconstruct the 15m window from question + end_time_est.

    window_end_utc == end_time_est (UTC, naive). window_start_utc = end - 15m.
    rules_ok=False if the question window does not resolve to end_time_est for
    either EDT or EST.
    """
    if end_time_est.tzinfo is not None:
        wend = end_time_est.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        wend = end_time_est
    offset = infer_et_offset(question, wend)
    wstart = wend - timedelta(minutes=WINDOW_MIN)
    if offset is None:
        return MarketWindow(
            market_id, asset, question, wstart, wend,
            et_offset_hours=None, rules_ok=False,
            rules_note="question-window/end_time_est mismatch",
        )
    return MarketWindow(
        market_id, asset, question, wstart, wend,
        et_offset_hours=offset, rules_ok=True,
    )