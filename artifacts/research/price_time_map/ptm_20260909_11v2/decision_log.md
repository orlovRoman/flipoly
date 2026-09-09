# Price-Time-Map Decision Log

Run: ptm_20260909_11v2
Rules: 7

## A_FUNNEL_FUNNEL_OBSERVED_SOL_YES_FAVORITE_[0.80,0.90)
- **Decision**: WATCHLIST_FOR_PAPER (NOT a significance-based candidate)
- **Evidence**: perm p=0.0005, Holm adj=0.1545 (>=0.05: NOT significant); CI=[0.035517, 0.187555], n=27, day_var=0.076136
- **Risk**: Nominal signal only; any PAPER bet requires new out-of-sample data.

## A_GRID_T-12_BTC_YES_FAVORITE_[0.90,0.99]
- **Decision**: WATCHLIST_FOR_PAPER (NOT a significance-based candidate)
- **Evidence**: perm p=0.0005, Holm adj=0.1545 (>=0.05: NOT significant); CI=[0.072158, 0.093785], n=20, day_var=0.001591
- **Risk**: Nominal signal only; any PAPER bet requires new out-of-sample data.

## A_GRID_T-5_BTC_YES_FAVORITE_ABOVE_0.99
- **Decision**: WATCHLIST_FOR_PAPER (NOT a significance-based candidate)
- **Evidence**: perm p=0.0005, Holm adj=0.1545 (>=0.05: NOT significant); CI=[0.002749, 0.004731], n=26, day_var=1.9e-05
- **Risk**: DEGENERATE micro-edge (day_var<1e-3): economically trivial even if real. Nominal signal only; any PAPER bet requires new out-of-sample data.

## C_T-12_vs_T-8_CT
- **Decision**: INFORMATIONAL_ONLY
- **Evidence**: mean_diff=-0.004629, CI=[-0.012062, 0.00513], perm_p=0.861069, markets=12744 (both_ok=4602)
- **Risk**: no pair significant after multiplicity (best Holm over 3 pairs = 0.10); earlier positive diffs were selection bias from conditioning on outsider status.

## C_T-12_vs_T-5_CT
- **Decision**: INFORMATIONAL_ONLY
- **Evidence**: mean_diff=0.010552, CI=[-0.013374, 0.048165], perm_p=0.114943, markets=12744 (both_ok=4525)
- **Risk**: no pair significant after multiplicity (best Holm over 3 pairs = 0.10); earlier positive diffs were selection bias from conditioning on outsider status.

## C_T-8_vs_T-5_CT
- **Decision**: INFORMATIONAL_ONLY
- **Evidence**: mean_diff=0.015181, CI=[-0.001639, 0.049772], perm_p=0.033483, markets=12744 (both_ok=6367)
- **Risk**: no pair significant after multiplicity (best Holm over 3 pairs = 0.10); earlier positive diffs were selection bias from conditioning on outsider status.

## D_CT_T5_ALL_ASSETS
- **Decision**: WATCHLIST_FOR_PAPER (pre-registered, NOT validated)
- **Evidence**: CT T-5 historical gross 100.103/915 (+0.109/entry); top5=195.0, ex_top5=-94.897, biggest_day=['2026-08-28', 92.49381199999999]
- **Risk**: high concentration (92% of pnl in one day); robustness and post-cost profit NOT established; this report neither proves profitability nor gives grounds to reject.
