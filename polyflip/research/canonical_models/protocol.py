"""Protocol load/validate/hash. Every result must embed protocol_hash."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:  # minimal fallback if pyyaml missing
    yaml = None

PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")

REQUIRED_TOP = [
    "protocol_version", "assets", "decision_point", "trading_universe",
    "forecast_variants", "periods", "challenger_selection", "economics",
    "final_status_criteria", "compute",
]


def _load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return dict(yaml.safe_load(text))
    # tiny fallback: only supports this file's flat subset -> require pyyaml
    raise RuntimeError("pyyaml is required to read protocol.yaml")


def load_protocol(path: Path | None = None) -> dict:
    p = Path(path) if path else PROTOCOL_PATH
    data = _load_yaml(p)
    missing = [k for k in REQUIRED_TOP if k not in data]
    if missing:
        raise ValueError(f"protocol.yaml missing keys: {missing}")
    if data["assets"] != ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
        raise ValueError("protocol assets must be exactly [BTC, ETH, SOL, XRP, DOGE]")
    dp = data["decision_point"]
    if dp.get("max_lateness_sec") != 15:
        raise ValueError("max_lateness_sec must be 15")
    tu = data["trading_universe"]
    if not (tu.get("ask_min_inclusive") == 0.01 and tu.get("ask_max_inclusive") == 0.40):
        raise ValueError("outsider ask range must be [0.01, 0.40] inclusive")
    if tu.get("budget_usd") != 1.0 or tu.get("budget_fee_included") is not False:
        raise ValueError("budget must be $1 purchase cost, fee accounted separately")
    if "PURCHASE COST" not in tu.get("budget_semantics", ""):
        raise ValueError("budget_semantics must pin purchase-cost semantics")
    return data


def canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def protocol_hash(protocol: dict | None = None) -> str:
    proto = protocol if protocol is not None else load_protocol()
    return hashlib.sha256(canonical_json(proto).encode("utf-8")).hexdigest()


def attach_hash(result: dict, protocol: dict | None = None) -> dict:
    out = dict(result)
    out["protocol_hash"] = protocol_hash(protocol)
    out["protocol_version"] = (protocol or load_protocol()).get("protocol_version")
    return out
