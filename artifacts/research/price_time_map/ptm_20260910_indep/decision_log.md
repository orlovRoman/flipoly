# Price-Time-Map Decision Log

Run: ptm_20260910_indep
Rules: 7

## A_FUNNEL_FUNNEL_OBSERVED_SOL_YES_FAVORITE_[0.80,0.90)
- **Decision**: WATCHLIST_FOR_PAPER (NOT a significance-based candidate)
- **Evidence**: perm p=0.0005, Holm adj=0.1555 (>=0.05: NOT significant); CI=[0.035517, 0.187555], n=27, day_var=0.076136
- **Risk**: Nominal signal only; any PAPER bet requires new out-of-sample data.

## A_GRID_T-5_BTC_YES_FAVORITE_ABOVE_0.99
- **Decision**: WATCHLIST_FOR_PAPER (NOT a significance-based candidate)
- **Evidence**: perm p=0.0005, Holm adj=0.1555 (>=0.05: NOT significant); CI=[0.002919, 0.004683], n=30, day_var=1.9e-05
- **Risk**: DEGENERATE micro-edge (day_var<1e-3): economically trivial even if real. Nominal signal only; any PAPER bet requires new out-of-sample data.

## A_GRID_T-5_DOGE_YES_FAVORITE_ABOVE_0.99
- **Decision**: WATCHLIST_FOR_PAPER (NOT a significance-based candidate)
- **Evidence**: perm p=0.0005, Holm adj=0.1555 (>=0.05: NOT significant); CI=[0.001117, 0.001696], n=32, day_var=5e-06
- **Risk**: DEGENERATE micro-edge (day_var<1e-3): economically trivial even if real. Nominal signal only; any PAPER bet requires new out-of-sample data.

## C_T-12_vs_T-8_CT
- **Decision**: INFORMATIONAL_ONLY
- **Evidence**: mean_diff=-0.007074, CI=[-0.012742, 0.004806], perm_p=0.969515, markets=14819 (both_ok=5728)
- **Risk**: no pair significant after multiplicity (best Holm over 3 pairs = 0.10); earlier positive diffs were selection bias from conditioning on outsider status.

## C_T-12_vs_T-5_CT
- **Decision**: INFORMATIONAL_ONLY
- **Evidence**: mean_diff=0.007297, CI=[-0.014516, 0.041186], perm_p=0.182909, markets=14819 (both_ok=5637)
- **Risk**: no pair significant after multiplicity (best Holm over 3 pairs = 0.10); earlier positive diffs were selection bias from conditioning on outsider status.

## C_T-8_vs_T-5_CT
- **Decision**: INFORMATIONAL_ONLY
- **Evidence**: mean_diff=0.01437, CI=[-0.000341, 0.044412], perm_p=0.018491, markets=14819 (both_ok=7919)
- **Risk**: no pair significant after multiplicity (best Holm over 3 pairs = 0.10); earlier positive diffs were selection bias from conditioning on outsider status.

## D_CT_T5_ALL_ASSETS
- **Decision**: WATCHLIST_FOR_PAPER (pre-registered, NOT validated)
- **Evidence**: CT T-5 historical gross 40.559/1115 (+0.109/entry); top5=195.0, ex_top5=-154.441, biggest_day=['2026-08-28', 90.49381199999999]
- **Risk**: high concentration (92% of pnl in one day); robustness and post-cost profit NOT established; this report neither proves profitability nor gives grounds to reject.
