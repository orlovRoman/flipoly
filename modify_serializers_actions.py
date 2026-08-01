def modify_serializers_actions():
    with open('polyflip/execution/serializers.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Add import
    import_stmt = "from polyflip.execution.states import RECONCILABLE_REQUEST_STATES\n"
    if "RECONCILABLE_REQUEST_STATES" not in content:
        content = import_stmt + content

    # Add field
    content = content.replace('"error_details": _parse_error(req.error_reason),', """
            "error_details": _parse_error(req.error_reason),
            "available_actions": ["RECONCILE_WITH_POLYMARKET"] if req.state in RECONCILABLE_REQUEST_STATES else [],
""")

    with open('polyflip/execution/serializers.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    modify_serializers_actions()
