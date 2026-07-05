import os

filepath = "tests/test_trainer_crypto.py"
with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

target_str = """        "consec_down":     np.random.randint(0, 5, n).astype(float),
    })"""
replacement_str = """        "consec_down":     np.random.randint(0, 5, n).astype(float),
        "target":          np.random.randint(0, 2, n),
    })"""

text = text.replace(target_str, replacement_str)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(text)
