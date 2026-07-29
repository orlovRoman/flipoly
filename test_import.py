import sys
print("starting import")
try:
    import polyflip.execution.settlement_service
    print("import success")
except Exception as e:
    print(f"error: {e}")
