from polyflip.trading.engine import kelly_bet_size

def test_kelly_bet_size_logic():
    # 1. Уверенный сигнал → ставка больше нуля
    bet = kelly_bet_size(p_win=0.70, buy_price=0.30, capital=1000)
    assert bet > 0, "При сильном сигнале ставка должна быть > 0"

    # 2. Неуверенный сигнал → ставка = 0
    bet_weak = kelly_bet_size(p_win=0.40, buy_price=0.50, capital=1000)
    assert bet_weak == 0.0, "При слабом сигнале (ожидаемый убыток) Kelly = 0"

    # 3. Ставка не превышает 10% капитала
    bet_max = kelly_bet_size(p_win=0.99, buy_price=0.01, capital=1000)
    assert bet_max <= 100.0, f"Ставка превышает 10% капитала: ${bet_max}"

    # 4. Граничная цена
    bet_edge = kelly_bet_size(p_win=0.80, buy_price=0.0, capital=1000)
    assert bet_edge == 0.0, "При нулевой цене Kelly должен вернуть 0"
