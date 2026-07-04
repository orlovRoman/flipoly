from polyflip.db.models import CryptoCandle, Base
from sqlalchemy import create_engine, inspect

engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)
insp = inspect(engine)
cols = {c["name"] for c in insp.get_columns("crypto_candles")}
required = {"symbol", "interval", "open_time", "open", "high", "low", "close", "volume", "taker_buy_volume"}
assert required <= cols, f"Отсутствуют колонки: {required - cols}"
print("✅ CryptoCandle ORM OK")
