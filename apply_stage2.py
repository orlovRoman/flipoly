import sys

with open("polyflip/research/stage2_regime_study.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add subprocess and hashlib imports
content = content.replace("import pandas as pd", "import pandas as pd\nimport subprocess\nimport hashlib")

# 2. Add git commit function inside run_stage2_study
git_func = """    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        git_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        git_commit = "UNKNOWN"
        git_dirty = False

    def hash_file(filepath):
        try:
            with open(filepath, 'rb') as f_in:
                return hashlib.sha256(f_in.read()).hexdigest()
        except:
            return "UNKNOWN"
            
    snaps_hash = hash_file(snapshots_csv_path)
    candles_hash = hash_file(candles_csv_path)
    exp_hash = hash_file(expirations_json_path)

    # 1. Load data"""
content = content.replace("    # 1. Load data", git_func)

# 3. Replace metadata git_commit
old_metadata = """"git_commit": "FIX_IN_PROGRESS","""
new_metadata = """"git_commit": git_commit,
            "git_dirty": git_dirty,
            "input_hashes": {
                "snapshots": snaps_hash,
                "candles": candles_hash,
                "expirations": exp_hash
            },
            "schema_version": "1.1.0","""
content = content.replace(old_metadata, new_metadata)

# 4. Modify price buckets
old_buckets = """    for p_low, p_high in price_buckets:
        b_key = f"[{p_low:.2f}, {p_high:.2f}]"
        mask_b = (c0_sub["executable_ask"] >= p_low) & (c0_sub["executable_ask"] < p_high)"""
new_buckets = """    for p_low, p_high in price_buckets:
        b_key = f"[{p_low:.2f}, {p_high:.2f}]"
        if p_high == 0.40:
            mask_b = (c0_sub["executable_ask"] >= p_low) & (c0_sub["executable_ask"] <= p_high)
        else:
            mask_b = (c0_sub["executable_ask"] >= p_low) & (c0_sub["executable_ask"] < p_high)"""
content = content.replace(old_buckets, new_buckets)

# 5. Modify exclusion rules
old_exclusions = """        # Observed quotes only: outsider is UP (yes_mid <= 0.5)
        if yes_mid > 0.5 or not np.isfinite(yes_ask):
            continue

        ask = yes_ask
        bid = yes_bid
        mid_p = yes_mid
        spread = float(dec_row["spread"]) if pd.notna(dec_row["spread"]) else (ask - (bid if np.isfinite(bid) else 0.0))
        final_outcome = str(dec_row["final_outcome"]).upper()

        # Primary rule: ask <= 0.40 and ask >= 0.01
        is_candidate_price = (ask <= 0.40) and (ask >= 0.01)
        if not is_candidate_price:
            continue

        # Item 19: Formal definition of Former Favorite (4 mutually exclusive groups)
        # Lookback window: up to 10 minutes prior to decision
        lb_cutoff = decision_at - pd.Timedelta(minutes=10)
        lb_snaps = group[(group["recorded_at"] >= lb_cutoff) & (group["recorded_at"] <= decision_at)]
        lb_prices = lb_snaps["mid_price"].dropna().tolist()

        if len(lb_prices) < 3:
            former_favorite_group = "INSUFFICIENT_HISTORY"
        else:
            max_lb = max(lb_prices)"""

new_exclusions = """        # Observed quotes only: outsider is UP (yes_mid <= 0.5)
        ask = yes_ask
        bid = yes_bid
        mid_p = yes_mid
        spread = float(dec_row["spread"]) if pd.notna(dec_row["spread"]) else (ask - (bid if np.isfinite(bid) else 0.0))
        final_outcome = str(dec_row["final_outcome"]).upper()
        
        exclusion_reason = None
        if not np.isfinite(yes_ask):
            exclusion_reason = "MISSING_QUOTE"
        elif yes_mid > 0.5:
            exclusion_reason = "PRICE_FILTER" # Outsider is UP
        
        # Primary rule: ask <= 0.40 and ask >= 0.01
        is_candidate_price = False
        if exclusion_reason == None:
            is_candidate_price = (ask <= 0.40) and (ask >= 0.01)
            if not is_candidate_price:
                exclusion_reason = "PRICE_FILTER"

        # Item 19: Formal definition of Former Favorite (4 mutually exclusive groups)
        # Lookback window: up to 10 minutes prior to decision
        lb_cutoff = decision_at - pd.Timedelta(minutes=10)
        lb_snaps = group[(group["recorded_at"] >= lb_cutoff) & (group["recorded_at"] <= decision_at)].copy()
        
        former_favorite_group = "INSUFFICIENT_HISTORY"
        if len(lb_snaps) >= 3:
            lb_times = lb_snaps["recorded_at"].tolist()
            duration = (lb_times[-1] - lb_times[0]).total_seconds()
            gaps = [ (lb_times[i] - lb_times[i-1]).total_seconds() for i in range(1, len(lb_times)) ]
            max_gap = max(gaps) if gaps else 0
            
            # minimal requirements (P17)
            if duration >= 300 and max_gap <= 300:
                lb_prices = lb_snaps["mid_price"].dropna().tolist()
                if len(lb_prices) >= 3:
                    max_lb = max(lb_prices)
                    if max_lb >= 0.55 and ask <= 0.40:
                        former_favorite_group = "FORMER_FAVORITE"
                    elif max_lb <= 0.40:
                        former_favorite_group = "PERSISTENT_CHEAP"
                    else:
                        former_favorite_group = "OTHER_TRAJECTORY"
"""
content = content.replace(old_exclusions, new_exclusions)

# 6. Add exclusion_reason to ledger row
content = content.replace('"execution_sim": "BLOCKED_DATA",', '"execution_sim": "BLOCKED_DATA",\n            "exclusion_reason": exclusion_reason,')

with open("polyflip/research/stage2_regime_study.py", "w", encoding="utf-8") as f:
    f.write(content)
