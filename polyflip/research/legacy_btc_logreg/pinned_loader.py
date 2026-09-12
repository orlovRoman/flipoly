"""Strict version-pinned loader for legacy LogReg artifacts (research only).

Why: ModelsCache.get(asset, type, version) silently falls back to the
current version, then to ANY same-asset entry, then to models[asset]
(ml_inference.py). For experiments such substitution is forbidden: a missing
version must raise, never return the current model.

Target semantics (verified, not assumed):
- legacy leaning LogRegs train on target=flip:
  target = ((mid_price > 0.5) != (final_outcome == "YES"))  (trainer.py),
  i.e. classes_ [0, 1] mean P(favourite loses) = P(outsider wins).
- mid == 0.5 excluded (outsider ambiguous).
- predict_proba[:, 1] is P(flip); flip_to_side_probs() maps it to P(UP)/P(DOWN).
"""
from __future__ import annotations

import hashlib
import pickle

EXPECTED_CLASSES = (0, 1)


class VersionNotPinnedError(LookupError):
    pass


def load_pinned(blob: bytes, expected_sha256: str, feature_order: list[str]):
    """Unpickle + verify hash + classes_. Never substitutes another version."""
    if not blob:
        raise VersionNotPinnedError("empty artifact bytes: refusing to substitute")
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected_sha256.lower().replace("sha256:", ""):
        raise VersionNotPinnedError(
            f"artifact hash mismatch: got {digest}, want {expected_sha256}")
    try:
        model = pickle.loads(blob)
    except Exception as exc:
        raise VersionNotPinnedError(f"unpickle failed: {exc}") from exc
    classes = tuple(int(c) for c in getattr(model, "classes_", ()))
    if classes != EXPECTED_CLASSES:
        raise VersionNotPinnedError(
            f"positive class is not flip==1: classes_={classes}")
    if not feature_order:
        raise VersionNotPinnedError("empty feature order")
    return model, list(feature_order)


def predict_p_flip(model, row: dict, feature_order: list[str]) -> float:
    """P(flip) for one ordered feature row. No calibration applied here."""
    import math
    x = [float(row[c]) for c in feature_order]
    if hasattr(model, "decision_function"):
        try:
            import pandas as pd
            z = float(model.decision_function(
                pd.DataFrame([x], columns=feature_order))[0])
        except ImportError:
            z = float(model.decision_function([x])[0])
        return 1.0 / (1.0 + math.exp(-z))
    return float(model.predict_proba([x])[0][1])


def flip_to_side_probs(mid_price: float, p_flip: float) -> dict:
    """Map P(flip) to side win probabilities. Parity (0.5) is an error."""
    if mid_price == 0.5:
        raise ValueError("parity mid==0.5: outsider undefined")
    if mid_price > 0.5:  # UP favourite -> outsider DOWN
        return {"p_up": 1.0 - p_flip, "p_down": p_flip, "outsider": "DOWN"}
    return {"p_up": p_flip, "p_down": 1.0 - p_flip, "outsider": "UP"}
