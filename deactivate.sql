UPDATE model_registry SET is_active = false WHERE asset IN (
    'BTCUSDT_low_vol', 'BTCUSDT_mid_vol', 'BTCUSDT_high_vol',
    'ETHUSDT_low_vol', 'ETHUSDT_mid_vol', 'ETHUSDT_high_vol',
    'XRPUSDT_low_vol', 'XRPUSDT_mid_vol', 'XRPUSDT_high_vol',
    'DOGEUSDT_low_vol', 'DOGEUSDT_mid_vol', 'DOGEUSDT_high_vol'
);
