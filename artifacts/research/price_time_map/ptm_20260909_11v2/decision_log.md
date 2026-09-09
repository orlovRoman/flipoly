# Price-Time-Map Decision Log

Run: ptm_20260909_11v2
Rules: 10

## A_GRID_T-12_BTC_YES_FAVORITE_[0.90,0.99]
- **Decision**: PROCEED_TO_HOLDOUT
- **Evidence**: Holm adj_p=0.0000, CI=[0.072158, 0.093785], n=20
- **Risk**: price_bin edge may not generalize; small n in some bins

## A_GRID_T-5_BTC_YES_FAVORITE_ABOVE_0.99
- **Decision**: PROCEED_TO_HOLDOUT
- **Evidence**: Holm adj_p=0.0000, CI=[0.002749, 0.004731], n=26
- **Risk**: price_bin edge may not generalize; small n in some bins

## A_GRID_T-5_DOGE_YES_FAVORITE_ABOVE_0.99
- **Decision**: PROCEED_TO_HOLDOUT
- **Evidence**: Holm adj_p=0.0000, CI=[0.001126, 0.001734], n=30
- **Risk**: price_bin edge may not generalize; small n in some bins

## A_GRID_T-5_ETH_YES_FAVORITE_ABOVE_0.99
- **Decision**: PROCEED_TO_HOLDOUT
- **Evidence**: Holm adj_p=0.0000, CI=[0.001877, 0.002723], n=37
- **Risk**: price_bin edge may not generalize; small n in some bins

## A_GRID_T-5_SOL_YES_FAVORITE_ABOVE_0.99
- **Decision**: PROCEED_TO_HOLDOUT
- **Evidence**: Holm adj_p=0.0000, CI=[0.001575, 0.002583], n=37
- **Risk**: price_bin edge may not generalize; small n in some bins

## A_GRID_T-5_XRP_YES_FAVORITE_ABOVE_0.99
- **Decision**: PROCEED_TO_HOLDOUT
- **Evidence**: Holm adj_p=0.0000, CI=[0.001419, 0.00271], n=29
- **Risk**: price_bin edge may not generalize; small n in some bins

## A_GRID_T-8_DOGE_YES_FAVORITE_[0.90,0.99]
- **Decision**: PROCEED_TO_HOLDOUT
- **Evidence**: Holm adj_p=0.0319, CI=[0.014513, 0.045932], n=218
- **Risk**: price_bin edge may not generalize; small n in some bins

## C_T-12_vs_T-8_CT
- **Decision**: INFORMATIONAL_ONLY
- **Evidence**: mean_diff=0.0464, markets=1682
- **Risk**: no CI provided; requires paired bootstrap for significance test

## C_T-12_vs_T-5_CT
- **Decision**: INFORMATIONAL_ONLY
- **Evidence**: mean_diff=0.1325, markets=1608
- **Risk**: no CI provided; requires paired bootstrap for significance test

## C_T-8_vs_T-5_CT
- **Decision**: INFORMATIONAL_ONLY
- **Evidence**: mean_diff=0.1225, markets=2633
- **Risk**: no CI provided; requires paired bootstrap for significance test
