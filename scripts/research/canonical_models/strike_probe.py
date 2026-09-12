"""Strike probe: local Gamma re-fetch for the 20 fixed markets (stdlib only).

Limit: 20 markets x (1 market + 1 event) calls, polite delay. No mass sweep,
no server compute. Saves raw JSON + request times for audit.
Out: artifacts/research/canonical_models/strike_probe/raw/<market_id>.{market,event}.json
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "artifacts" / "research" / "canonical_models" / "strike_probe"
RAW = PROBE / "raw"
GAMMA = "https://gamma-api.polymarket.com"
UA = {"User-Agent": "canonical-strike-probe/1"}


def get(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    RAW.mkdir(parents=True, exist_ok=True)
    markets = [
        line.split("|")[0]
        for line in (PROBE / "markets.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ][:limit]
    log = []
    for mid in markets:
        t0 = datetime.now(timezone.utc).isoformat()
        try:
            m = get(f"{GAMMA}/markets/{mid}")
        except Exception as e:  # noqa: BLE001 - probe must record, not crash
            log.append({"market_id": mid, "at": t0, "error": str(e)})
            continue
        (RAW / f"{mid}.market.json").write_text(json.dumps(m, indent=1))
        time.sleep(0.5)
        # event linkage (field names vary; record whatever exists)
        ev = {k: m.get(k) for k in ("eventId", "eventSlug", "events", "event") if m.get(k) is not None}
        entry: dict = {"market_id": mid, "at": t0, "closed": m.get("closed"),
                       "event_link": ev, "event": None}
        for key in ("eventId",):
            if m.get(key):
                try:
                    entry["event"] = get(f"{GAMMA}/events/{m[key]}")
                    (RAW / f"{mid}.event.json").write_text(json.dumps(entry["event"], indent=1))
                except Exception as e:  # noqa: BLE001
                    entry["event_error"] = str(e)
                time.sleep(0.5)
                break
        log.append(entry)
        print(f"{mid} closed={m.get('closed')} event_keys={sorted(ev)}")
    (PROBE / "fetch_log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {len(log)} entries -> {RAW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
