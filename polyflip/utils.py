

def parse_float_setting(raw: dict, key: str, default: float = 0.0) -> float:
    val = raw.get(key)
    if val is None or str(val).strip() == '':
        return default
    try:
        return float(val)
    except ValueError:
        return default
