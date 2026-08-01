import re

def modify_api():
    with open('polyflip/api/execution_api.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Insert readiness calculation before return
    readiness_logic = """
    readiness = None
    if active_session:
        readiness = await evaluate_live_readiness(db, active_session)
        
    status = active_session.status if active_session else None
    active_pos_count = len([p for p in active_positions if p.position_status not in ("CLOSED", "RESOLVED", "ENTRY_FAILED")])

    session_actions = {
        "check_readiness": status in {"DRAFT", "READY", "STOPPED"},
        "activate": status == "READY",
        "stop": status == "ACTIVE",
        "close_all": active_pos_count > 0,
        "finish": status in {"DRAFT", "READY", "STOPPED"},
    } if status else {
        "check_readiness": False,
        "activate": False,
        "stop": False,
        "close_all": False,
        "finish": False,
    }

    return {
        "available_actions": session_actions,
        "readiness": {
            "ready": readiness.ready,
            "checks": readiness.checks,
            "errors": readiness.errors,
            "warnings": readiness.warnings,
        } if readiness else None,"""
    
    content = re.sub(r'    return \{\s+"session": \(', readiness_logic + '\n        "session": (', content, count=1)

    with open('polyflip/api/execution_api.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    modify_api()
