import re

def modify_serializers_errors():
    with open('polyflip/execution/serializers.py', 'r', encoding='utf-8') as f:
        content = f.read()

    error_parser = """
def _parse_error(error_reason: str | None) -> dict:
    if not error_reason:
        return {"error_code": None, "error_message_ru": None}
    
    normalized = error_reason.lower()
    if "insufficient funds" in normalized:
        return {"error_code": "INSUFFICIENT_FUNDS", "error_message_ru": "Недостаточно средств на балансе или allowance"}
    if "minimum order size" in normalized or "below minimum" in normalized:
        return {"error_code": "ORDER_BELOW_MINIMUM", "error_message_ru": "Сумма заявки меньше минимальной суммы Polymarket ($0.50)"}
    if "max_slippage" in normalized or "slippage" in normalized:
        return {"error_code": "SLIPPAGE_EXCEEDED", "error_message_ru": "Превышено допустимое проскальзывание (slippage)"}
    if "market closed" in normalized or "market is closed" in normalized:
        return {"error_code": "MARKET_CLOSED", "error_message_ru": "Рынок уже закрыт или разрешен"}
        
    return {"error_code": "UNKNOWN_ERROR", "error_message_ru": error_reason}
"""
    
    content = content.replace("async def serialize_execution_requests(", error_parser + "\nasync def serialize_execution_requests(")
    
    content = content.replace('"error_reason": req.error_reason,', """
            "error_reason": req.error_reason,
            "error_details": _parse_error(req.error_reason),
""")

    with open('polyflip/execution/serializers.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    modify_serializers_errors()
