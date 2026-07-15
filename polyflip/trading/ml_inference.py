import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any
import structlog
from dataclasses import dataclass

logger = structlog.get_logger(__name__)


@dataclass
class ModelsCache:
    models: dict[str, Any]
    versions: dict[str, int]
    features: dict[str, list[str]]


def build_inference_dataframe(
    market: Any,
    history_snaps: list[Any],
    fresh_yes_price: float,
    fresh_spread: float,
    global_max: float,
    start_time: datetime,
    time_left_sec: float,
) -> pd.DataFrame:
    """
    Строит DataFrame для инференса модели на основе исторических снапшотов и текущих (свежих) данных.
    """
    rows = []
    for snap in history_snaps:
        rows.append({
            "time_left_min": getattr(snap, "time_left_min", 0.0),
            "mid_price": getattr(snap, "mid_price", 0.0),
            "spread": getattr(snap, "spread", 0.0),
            "price_velocity": getattr(snap, "price_velocity", 0.0),
            "volume_5min": getattr(snap, "volume_5min", 0.0),
            "hour_of_day": getattr(snap, "hour_of_day", 0),
            "market_id": getattr(snap, "market_id", ""),
            "recorded_at": getattr(snap, "recorded_at", None),
        })
        
    rows.append({
        "time_left_min": time_left_sec / 60.0,
        "mid_price": fresh_yes_price,
        "spread": fresh_spread,
        "price_velocity": getattr(market, "price_velocity", 0.0) or 0.0,
        "volume_5min": getattr(market, "volume_5min", 0.0) or 0.0,
        "hour_of_day": start_time.hour,
        "market_id": getattr(market, "market_id", ""),
        "recorded_at": start_time,
    })
    
    df = pd.DataFrame(rows)
    df["price_distance_from_max"] = (global_max - df["mid_price"]).clip(lower=0.0)
    
    if "recorded_at" in df.columns:
        df = df.drop(columns=["recorded_at"])
    if "market_id" in df.columns:
        df = df.drop(columns=["market_id"])
        
    return df


def run_model_inference(
    df: pd.DataFrame,
    model: Any,
    features: list[str],
) -> float:
    """
    Прогоняет DataFrame через модель и возвращает вероятность для класса 1 (flip).
    Если модель возвращает только один класс, возвращает 0.0.
    """
    X = df[features]
    proba = model.predict_proba(X)
    
    try:
        p_flip = float(proba[-1][1])
    except IndexError:
        p_flip = 0.0
        
    return p_flip
