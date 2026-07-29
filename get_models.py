from sqlalchemy import create_engine
import pandas as pd
engine = create_engine("postgresql://polyflip:polyflip@localhost:5432/polyflip")
df = pd.read_sql("SELECT asset, version, is_active, accuracy, ece, baseline FROM model_registry WHERE asset LIKE '%%USDT%%' ORDER BY asset, version DESC", engine)
print(df.to_string(index=False))
