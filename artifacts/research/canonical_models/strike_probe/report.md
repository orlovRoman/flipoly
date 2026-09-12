# Strike probe report (Gamma re-fetch, 20 markets)

Date: 2026-09-10. List fixed pre-request (`markets.txt`, committed a973399).
Method: `scripts/research/canonical_models/strike_probe.py`, 20 markets ×
1 market call (event linkage absent → 0 event calls), polite delay.
Raw: `raw/*.market.json` (68 KB) + `fetch_log.json` (request times, closed flags).

## Checked fields (explicit numeric opening / price-to-beat of the contract)
`strike, strikePrice, priceToBeat, openingPrice, underlying_price,
underlyingPrice, startPrice, openPrice, referencePrice, anchorPrice`,
plus any numeric `*price*` field.

## Result
- 20/20 markets `closed=true`, all 5 assets, ends 2026-07-01 → 2026-09-09.
- 0/20 contain any strike-like field. Only token-price stats
  (`lastTradePrice`, `oneDay/HourPriceChange` — outcome-token prices,
  NOT underlying strike). No event-linkage fields on market objects.
- Rules text (`description`) + Chainlink stream URL confirm the family rule
  (TWAP ≥ window-start → UP) but are NOT the strike value itself.

## Verdict
`canonical_strike_status (Gamma path) = DATA_BLOCKED`.
Reason: per-contract numeric strike is absent from Gamma market objects
(post-close included); meaning of any substitute unestablished.
No Binance open/close substitution applied.
Next path (per plan): Chainlink history ONLY after confirming the exact
feed, history availability, and contract calc rules. Access/subscription
not assumed.
