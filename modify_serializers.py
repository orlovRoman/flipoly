def modify_serializers():
    with open('polyflip/execution/serializers.py', 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace('"asset": req.asset,', '"asset": req.asset,\n            "outcome_to_buy": req.outcome_to_buy,')

    with open('polyflip/execution/serializers.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    modify_serializers()
