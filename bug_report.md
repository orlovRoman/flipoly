# Отчет о найденных багах в репозитории PolyFlip

## 1. Логические ошибки и ошибки типизации (Mypy)
Найдено 301 ошибок типизации, не связанных с отсутствием стабов.

### Файл: `polyflip/execution/states.py`
- 82: error: Name "StrEnum" already defined (possibly by an import)  [no-redef]

### Файл: `polyflip/trading/weighted_policy.py`
- 871: error: Argument 9 to "replace" of "WeightedPolicyConfig" has incompatible type "**dict[str, object]"; expected "float"  [arg-type]
- 886: error: Argument 2 to "replace" of "WeightedPolicyConfig" has incompatible type "**dict[str, object]"; expected "float"  [arg-type]
- 876: error: Argument 9 to "replace" of "WeightedPolicyConfig" has incompatible type "**dict[str, object]"; expected "float"  [arg-type]
- 881: error: Argument 9 to "replace" of "WeightedPolicyConfig" has incompatible type "**dict[str, object]"; expected "tuple[tuple[str, tuple[float, ...]], ...]"  [arg-type]
- 886: error: Argument 2 to "replace" of "WeightedPolicyConfig" has incompatible type "**dict[str, object]"; expected "tuple[tuple[str, tuple[float, ...]], ...]"  [arg-type]
- 876: error: Argument 9 to "replace" of "WeightedPolicyConfig" has incompatible type "**dict[str, object]"; expected "tuple[tuple[str, tuple[float, ...]], ...]"  [arg-type]
- 288: error: Incompatible types in assignment (expression has type "float | None", variable has type "float")  [assignment]
- 881: error: Argument 9 to "replace" of "WeightedPolicyConfig" has incompatible type "**dict[str, object]"; expected "str"  [arg-type]
- 876: error: Argument 9 to "replace" of "WeightedPolicyConfig" has incompatible type "**dict[str, object]"; expected "str"  [arg-type]
- 884: error: Argument 4 to "replace" of "WeightedPolicyConfig" has incompatible type "**dict[str, object]"; expected "tuple[str, ...]"  [arg-type]
- *...и еще 10 ошибок*

### Файл: `polyflip/execution/assets.py`
- 39: error: No overload variant of "list" matches argument type "object"  [call-overload]

### Файл: `polyflip/crypto/feature_builder.py`
- 216: error: Argument 1 to "float" has incompatible type "Any | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]

### Файл: `polyflip/db/models.py`
- 669: error: Variable "polyflip.db.models.Base" is not valid as a type  [valid-type]
- 1172: error: Invalid base class "Base"  [misc]
- 905: error: Variable "polyflip.db.models.Base" is not valid as a type  [valid-type]
- 669: error: Invalid base class "Base"  [misc]
- 1246: error: Invalid base class "Base"  [misc]
- 1803: error: Invalid base class "Base"  [misc]
- 1585: error: Variable "polyflip.db.models.Base" is not valid as a type  [valid-type]
- 1662: error: Variable "polyflip.db.models.Base" is not valid as a type  [valid-type]
- 1702: error: Variable "polyflip.db.models.Base" is not valid as a type  [valid-type]
- 1742: error: Invalid base class "Base"  [misc]
- *...и еще 72 ошибок*

### Файл: `polyflip/trading/weighted_sizing.py`
- 79: error: Unsupported operand types for * ("float" and "None")  [operator]

### Файл: `polyflip/ai_lab/llm.py`
- 837: error: Name "body" already defined on line 821  [no-redef]
- 975: error: Name "body" already defined on line 968  [no-redef]
- 1066: error: Argument "model_summary" to "OpenAIResponsesProvider" has incompatible type "str | Any | None"; expected "str"  [arg-type]
- 1065: error: Argument "model_research" to "OpenAIResponsesProvider" has incompatible type "str | Any | None"; expected "str"  [arg-type]

### Файл: `polyflip/crypto/polymarket_backtest.py`
- 116: error: Argument 1 to "float" has incompatible type "Iterable[float] | float"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
- 118: error: Argument 1 to "list" has incompatible type "Iterable[float] | float"; expected "Iterable[float]"  [arg-type]
- 725: error: Need type annotation for "coverage_reasons"  [var-annotated]
- 411: error: Need type annotation for "coverage_reasons"  [var-annotated]
- 742: error: Argument 1 to "float" has incompatible type "Any | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
- 452: error: Argument "config" to "select_weighted_side" has incompatible type "WeightedPolicyConfig | None"; expected "WeightedPolicyConfig"  [arg-type]
- 541: error: Argument "fee_rate" to "_pnl_for_trade" has incompatible type "float | None"; expected "float"  [arg-type]
- 390: error: Argument 1 to "float" has incompatible type "float | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
- 748: error: Argument 1 to "float" has incompatible type "Any | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]

### Файл: `polyflip/trading/weighted_benchmark.py`
- 2207: error: Argument 8 to "evaluate_arm" has incompatible type "**dict[str, object]"; expected "tuple[float, float] | None"  [arg-type]
- 762: error: Argument 4 to "WeightedPolicyConfig" has incompatible type "**dict[str, object]"; expected "tuple[str, ...]"  [arg-type]
- 2251: error: Argument 7 to "stability_by_segment" has incompatible type "**dict[str, object]"; expected "str"  [arg-type]
- 2207: error: Argument 8 to "evaluate_arm" has incompatible type "**dict[str, object]"; expected "bool"  [arg-type]
- 1669: error: Argument 2 to "replace" of "WeightedPolicyConfig" has incompatible type "**dict[str, float]"; expected "str"  [arg-type]
- 2224: error: Argument 6 to "evaluate_arm" has incompatible type "**dict[str, object]"; expected "tuple[float, float] | None"  [arg-type]
- 762: error: Argument 4 to "WeightedPolicyConfig" has incompatible type "**dict[str, object]"; expected "tuple[tuple[str, tuple[float, ...]], ...]"  [arg-type]
- 758: error: Argument 4 to "WeightedPolicyConfig" has incompatible type "**dict[str, object]"; expected "tuple[float, ...]"  [arg-type]
- 2251: error: Argument 7 to "stability_by_segment" has incompatible type "**dict[str, object]"; expected "float"  [arg-type]
- 2241: error: Argument 5 to "parameter_sensitivity" has incompatible type "**dict[str, object]"; expected "str"  [arg-type]
- *...и еще 38 ошибок*

### Файл: `polyflip/trading/policy_artifact.py`
- 363: error: Unsupported operand types for <= ("float" and "None")  [operator]
- 345: error: Unsupported operand types for - ("None" and "float")  [operator]
- 345: error: Unsupported operand types for - ("float" and "None")  [operator]
- 343: error: Unsupported left operand type for - ("None")  [operator]
- 361: error: Unsupported left operand type for <= ("None")  [operator]
- 361: error: Unsupported operand types for <= ("float" and "None")  [operator]
- 363: error: Unsupported operand types for >= ("float" and "None")  [operator]
- 361: error: Unsupported operand types for >= ("float" and "None")  [operator]
- 343: error: Unsupported operand types for - ("float" and "None")  [operator]
- 345: error: Unsupported left operand type for - ("None")  [operator]
- *...и еще 2 ошибок*

### Файл: `polyflip/crypto/experiment_configs.py`
- 233: error: Incompatible types in assignment (expression has type "str", target has type "int | float")  [assignment]
- 234: error: Incompatible types in assignment (expression has type "str", target has type "int | float")  [assignment]

### Файл: `polyflip/ai_lab/llm_catalog.py`
- 285: error: Item "None" of "Any | None" has no attribute "protocol"  [union-attr]
- 291: error: Item "None" of "Any | None" has no attribute "probe_status"  [union-attr]
- 287: error: Item "None" of "Any | None" has no attribute "is_available"  [union-attr]
- 289: error: Item "None" of "Any | None" has no attribute "is_discovered"  [union-attr]
- 288: error: Item "None" of "Any | None" has no attribute "raw_metadata"  [union-attr]
- 537: error: Incompatible return value type (got "dict[str, AILLMModelCatalog]", expected "dict[str, dict[str, Any]]")  [return-value]
- 292: error: Item "None" of "Any | None" has no attribute "expires_at"  [union-attr]
- 286: error: Item "None" of "Any | None" has no attribute "supports_structured_output"  [union-attr]
- 202: error: Item "None" of "datetime | None" has no attribute "isoformat"  [union-attr]
- 284: error: Item "None" of "Any | None" has no attribute "display_name"  [union-attr]
- *...и еще 1 ошибок*

### Файл: `polyflip/crypto/threshold_optimizer.py`
- 449: error: No overload variant of "__setitem__" of "list" matches argument types "int", "dict[str, Any]"  [call-overload]
- 444: error: No overload variant of "__setitem__" of "list" matches argument types "int", "dict[str, Any]"  [call-overload]

### Файл: `polyflip/crypto/market_regime_policy.py`
- 152: error: Argument 1 to "float" has incompatible type "object"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]

### Файл: `polyflip/ai_lab/service.py`
- 1020: error: Incompatible return value type (got "tuple[Any, Any]", expected "DeploymentRevision")  [return-value]

### Файл: `polyflip/api/mrf_api.py`
- 175: error: Argument 1 to "int" has incompatible type "Any | None"; expected "str | Buffer | SupportsInt | SupportsIndex | SupportsTrunc"  [arg-type]
- 111: error: Item "None" of "Any | dict[Any, Any] | None" has no attribute "get"  [union-attr]
- 102: error: Item "None" of "Any | dict[Any, Any] | None" has no attribute "get"  [union-attr]
- 311: error: Need type annotation for "asset_stats" (hint: "asset_stats: dict[<type>, <type>] = ...")  [var-annotated]
- 108: error: Item "None" of "Any | dict[Any, Any] | None" has no attribute "get"  [union-attr]
- 114: error: Item "None" of "Any | dict[Any, Any] | None" has no attribute "get"  [union-attr]
- 298: error: Need type annotation for "phase_counts" (hint: "phase_counts: dict[<type>, <type>] = ...")  [var-annotated]
- 105: error: Item "None" of "Any | dict[Any, Any] | None" has no attribute "get"  [union-attr]

### Файл: `polyflip/ai_lab/executor.py`
- 404: error: Argument "slices" to "record_result" has incompatible type "Mapping[str, Any]"; expected "dict[str, Any] | None"  [arg-type]
- 400: error: Argument "config_id" to "record_result" has incompatible type "int | None"; expected "int"  [arg-type]
- 403: error: Argument "metrics" to "record_result" has incompatible type "Mapping[str, Any]"; expected "dict[str, Any] | None"  [arg-type]
- 354: error: Incompatible types in assignment (expression has type "SimpleNamespace", variable has type "AIExperimentJob")  [assignment]
- 316: error: Argument "config_id" to "StepContext" has incompatible type "int | None"; expected "int"  [arg-type]

### Файл: `polyflip/api/ai_lab_agent.py`
- 1104: error: Argument 1 to "int" has incompatible type "int | None"; expected "str | Buffer | SupportsInt | SupportsIndex | SupportsTrunc"  [arg-type]
- 1087: error: Argument 1 to "int" has incompatible type "int | None"; expected "str | Buffer | SupportsInt | SupportsIndex | SupportsTrunc"  [arg-type]
- 1109: error: Argument 1 to "int" has incompatible type "int | None"; expected "str | Buffer | SupportsInt | SupportsIndex | SupportsTrunc"  [arg-type]
- 650: error: Argument 1 to "dict" has incompatible type "Any | None"; expected "SupportsKeysAndGetItem[str, Any]"  [arg-type]

### Файл: `polyflip/models/trainer.py`
- 655: error: Need type annotation for "status_messages" (hint: "status_messages: dict[<type>, <type>] = ...")  [var-annotated]
- 363: error: Argument "key" to "max" has incompatible type overloaded function; expected "Callable[[float], SupportsDunderLT[Any] | SupportsDunderGT[Any]]"  [arg-type]

### Файл: `polyflip/execution/order_strategies.py`
- 355: error: Incompatible types in assignment (expression has type "str | None", variable has type "str")  [assignment]
- 833: error: Name "updates" already defined on line 776  [no-redef]

### Файл: `polyflip/execution/gateways/polymarket.py`
- 511: error: Need type annotation for "allowances" (hint: "allowances: dict[<type>, <type>] = ...")  [var-annotated]

### Файл: `polyflip/collector/client.py`
- 182: error: Argument 1 to "float" has incompatible type "Any | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
- 42: error: Argument 1 to "float" has incompatible type "Any | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
- 192: error: Argument 1 to "float" has incompatible type "Any | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]

### Файл: `polyflip/api/trading_dashboard.py`
- 48: error: Need type annotation for "_stats_cache" (hint: "_stats_cache: dict[<type>, <type>] = ...")  [var-annotated]

### Файл: `polyflip/api/dashboard.py`
- 1225: error: Need type annotation for "exact_trades_count"  [var-annotated]
- 1144: error: Unsupported operand types for - ("float" and "object")  [operator]
- 497: error: Invalid index type "str" for "dict[int, dict[str, object]]"; expected type "int"  [index]
- 495: error: Unsupported operand types for - ("float" and "object")  [operator]
- 358: error: Need type annotation for "_model_pnl_cache" (hint: "_model_pnl_cache: dict[<type>, <type>] = ...")  [var-annotated]
- 293: error: Need type annotation for "activity_data" (hint: "activity_data: dict[<type>, <type>] = ...")  [var-annotated]
- 159: error: Need type annotation for "_dashboard_cache" (hint: "_dashboard_cache: dict[<type>, <type>] = ...")  [var-annotated]
- 655: error: Invalid index type "str" for "dict[int, dict[str, object]]"; expected type "int"  [index]
- 713: error: Need type annotation for "execution_failure_map" (hint: "execution_failure_map: dict[<type>, <type>] = ...")  [var-annotated]
- 1226: error: Need type annotation for "reconstructed_trades_count"  [var-annotated]
- *...и еще 3 ошибок*

### Файл: `polyflip/api/analytics.py`
- 160: error: Need type annotation for "_models_cache" (hint: "_models_cache: dict[<type>, <type>] = ...")  [var-annotated]
- 251: error: Incompatible types in assignment (expression has type "list[dict[str, Any]]", target has type "float")  [assignment]
- 449: error: Incompatible default for argument "last_run" (default has type "None", argument has type "str")  [assignment]

### Файл: `polyflip/api/execution_api.py`
- 486: error: Need type annotation for "result"  [var-annotated]
- 1566: error: Argument 1 to "get" of "dict" has incompatible type "str | None"; expected "str"  [arg-type]
- 1621: error: Incompatible types in assignment (expression has type "None", variable has type "SessionBudgetSnapshot")  [assignment]

### Файл: `polyflip/crypto/trainer.py`
- 1145: error: Need type annotation for "backtest_variants"  [var-annotated]
- 1442: error: Argument 1 to "predict" of "CryptoPredictor" has incompatible type "Sequence[CryptoCandle]"; expected "list[Any]"  [arg-type]
- 1332: error: Item "None" of "list[dict[str, object]] | dict[str, dict[str, float | None]] | str | dict[str, object] | dict[str, dict[str, float | int]] | float | None" has no attribute "get"  [union-attr]
- 360: error: Incompatible types in assignment (expression has type "tuple[str, str]", variable has type "tuple[str]")  [assignment]
- 1332: error: Item "str" of "list[dict[str, object]] | dict[str, dict[str, float | None]] | str | dict[str, object] | dict[str, dict[str, float | int]] | float | None" has no attribute "get"  [union-attr]
- 1332: error: Item "float" of "list[dict[str, object]] | dict[str, dict[str, float | None]] | str | dict[str, object] | dict[str, dict[str, float | int]] | float | None" has no attribute "get"  [union-attr]
- 1332: error: Item "list[dict[str, object]]" of "list[dict[str, object]] | dict[str, dict[str, float | None]] | str | dict[str, object] | dict[str, dict[str, float | int]] | float | None" has no attribute "get"  [union-attr]
- 736: error: Argument 2 to "compute_oof_polymarket_backtest" has incompatible type "Any | None"; expected "Iterable[float]"  [arg-type]

### Файл: `polyflip/crypto/market_direction_service.py`
- 111: error: Argument 1 to "predict" of "CryptoPredictor" has incompatible type "Sequence[CryptoCandle]"; expected "list[Any]"  [arg-type]

### Файл: `polyflip/crypto/candle_collector.py`
- 73: error: Unexpected keyword argument "symbol" for "warning" of "Logger"  [call-arg]
- 82: error: Unexpected keyword argument "symbol" for "info" of "Logger"  [call-arg]
- 77: error: Unexpected keyword argument "symbol" for "warning" of "Logger"  [call-arg]
- 77: error: Unexpected keyword argument "interval" for "warning" of "Logger"  [call-arg]
- 82: error: Unexpected keyword argument "interval" for "info" of "Logger"  [call-arg]
- 73: error: Unexpected keyword argument "interval" for "warning" of "Logger"  [call-arg]
- 77: error: Unexpected keyword argument "error" for "warning" of "Logger"  [call-arg]
- 82: error: Unexpected keyword argument "inserted" for "info" of "Logger"  [call-arg]

### Файл: `polyflip/execution/gateways/factory.py`
- 47: error: Argument "relayer_api_key_address" to "PolymarketExecutionGateway" has incompatible type "str | None"; expected "str"  [arg-type]

### Файл: `polyflip/crypto/backtester.py`
- 451: error: Need type annotation for "pnl_curve" (hint: "pnl_curve: list[<type>] = ...")  [var-annotated]
- 260: error: Item "None" of "Any | None" has no attribute "predict_proba"  [union-attr]

### Файл: `polyflip/ai_lab/lgbm_adapters.py`
- 612: error: "Mapping[Any, Any]" has no attribute "setdefault"  [attr-defined]
- 613: error: "Mapping[Any, Any]" has no attribute "setdefault"  [attr-defined]

### Файл: `polyflip/execution/worker.py`
- 1261: error: Return value expected  [return-value]
- 206: error: Argument 3 to "FAKRetryEdgePolicy" has incompatible type "**dict[str, Decimal | None]"; expected "Decimal"  [arg-type]
- 1335: error: Return value expected  [return-value]
- 2041: error: Trying to read deleted variable "exc"  [misc]

### Файл: `polyflip/trading/combined_voting.py`
- 1217: error: Argument 1 to "round" has incompatible type "float | None"; expected "_SupportsRound2[float]"  [arg-type]
- 739: error: Argument 1 to "float" has incompatible type "Any | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
- 209: error: Argument 1 to "float" has incompatible type "Any | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
- 825: error: Argument 1 to "float" has incompatible type "Any | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
- 2063: error: Argument 2 to "_weighted_sizing_fields" has incompatible type "WeightedSelection | None"; expected "WeightedSelection"  [arg-type]
- 1222: error: Incompatible types in assignment (expression has type "float | None", variable has type "float")  [assignment]

### Файл: `polyflip/api/crypto_dashboard.py`
- 905: error: Unsupported operand types for - ("None" and "float")  [operator]
- 904: error: Unsupported operand types for - ("None" and "float")  [operator]
- 905: error: Unsupported operand types for - ("float" and "None")  [operator]
- 905: error: Unsupported left operand type for - ("None")  [operator]
- 904: error: Unsupported operand types for - ("float" and "None")  [operator]
- 1613: error: Incompatible types in assignment (expression has type "datetime", target has type "str")  [assignment]
- 1545: error: Incompatible default for argument "date_to" (default has type "None", argument has type "str")  [assignment]
- 1544: error: Incompatible default for argument "date_from" (default has type "None", argument has type "str")  [assignment]
- 904: error: Unsupported left operand type for - ("None")  [operator]
- 906: error: Unsupported operand types for - ("None" and "float")  [operator]
- *...и еще 6 ошибок*

### Файл: `polyflip/trading/trade_recorder.py`
- 282: error: Incompatible types in assignment (expression has type "dict[str, str | float | Any | None]", target has type "str")  [assignment]

### Файл: `polyflip/trading/decision_runners.py`
- 323: error: Argument 1 to "build_snapshot_from_multi_asset_candles" has incompatible type "dict[str, Sequence[CryptoCandle] | BaseException]"; expected "dict[str, Sequence[Any]]"  [arg-type]
- 1069: error: Argument "strategy_type" to "TradeDecision" has incompatible type "Literal['COMBINED', 'LOGREG_ONLY']"; expected "Literal['OUTSIDER', 'COMBINED', 'SKIP']"  [arg-type]
- 1053: error: Argument "strategy_type" to "TradeDecision" has incompatible type "Literal['COMBINED', 'LOGREG_ONLY']"; expected "Literal['OUTSIDER', 'COMBINED', 'SKIP']"  [arg-type]
- 551: error: Name "decide_ml_mode" is not defined  [name-defined]
- 38: error: Argument 1 to "float" has incompatible type "Any | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]

### Файл: `polyflip/backtesting/runner.py`
- 182: error: Item "None" of "MarketTick | None" has no attribute "recorded_at"  [union-attr]

### Файл: `polyflip/api/backtest_api.py`
- 209: error: Incompatible types in assignment (expression has type "datetime", variable has type "None")  [assignment]
- 207: error: Incompatible types in assignment (expression has type "str", variable has type "None")  [assignment]

### Файл: `polyflip/api/ai_lab.py`
- 804: error: Item "None" of "Any | dict[Any, Any] | None" has no attribute "get"  [union-attr]
- 805: error: Item "None" of "Any | dict[Any, Any] | None" has no attribute "get"  [union-attr]

### Файл: `polyflip/api/main.py`
- 41: error: Need type annotation for "requests"  [var-annotated]


## 2. Синтаксические и стилевые ошибки (Flake8)
Найдено 1149 ошибок (исключая ошибки длины строки E501).

### Файл: `polyflip/ai_lab/agent.py`
- 59:1: F401 'polyflip.db.models.ModelRegistry' imported but unused
- 59:1: F401 'polyflip.db.models.ExperimentResult' imported but unused
- 59:1: F401 'polyflip.db.models.AIConfigOverlay' imported but unused
- 59:1: F401 'polyflip.db.models.AIShadowAssignment' imported but unused
- 25:1: F401 'polyflip.ai_lab.llm.HypothesisProposal' imported but unused
- *...и еще 9 ошибок*

### Файл: `polyflip/ai_lab/agent_tools.py`
- 20:1: F401 'polyflip.ai_lab.orchestrator.promote_to_shadow' imported but unused
- 32:1: F401 'polyflip.db.models.AIOptimizationRun' imported but unused
- 21:1: F401 'polyflip.ai_lab.service.transition_run' imported but unused
- 21:1: F401 'polyflip.ai_lab.service.propose_live_deployment' imported but unused
- 21:1: F401 'polyflip.ai_lab.service.append_step' imported but unused
- *...и еще 12 ошибок*

### Файл: `polyflip/ai_lab/lgbm_adapters.py`
- 12:1: F401 'json' imported but unused
- 11:1: F401 'hashlib' imported but unused

### Файл: `polyflip/ai_lab/lgbm_worker.py`
- 14:1: E402 module level import not at top of file

### Файл: `polyflip/ai_lab/llm.py`
- 633:5: E303 too many blank lines (2)
- 316:1: E302 expected 2 blank lines, found 1

### Файл: `polyflip/ai_lab/llm_catalog.py`
- 1021:9: E301 expected 1 blank line, found 0

### Файл: `polyflip/ai_lab/logreg_adapters.py`
- 10:1: F401 'json' imported but unused
- 9:1: F401 'hashlib' imported but unused
- 19:1: F401 'polyflip.ai_lab.executor.ACTION_TO_EVALUATION_KIND' imported but unused

### Файл: `polyflip/ai_lab/orchestrator.py`
- 376:1: E302 expected 2 blank lines, found 1
- 227:1: E303 too many blank lines (3)
- 183:1: E303 too many blank lines (3)
- 293:1: E302 expected 2 blank lines, found 0

### Файл: `polyflip/ai_lab/scheduler.py`
- 15:1: F401 'typing.Any' imported but unused

### Файл: `polyflip/ai_lab/thread_provider.py`
- 67:1: E305 expected 2 blank lines after class or function definition, found 1

### Файл: `polyflip/api/ai_lab.py`
- 482:1: E302 expected 2 blank lines, found 0
- 905:1: E302 expected 2 blank lines, found 0
- 39:1: F401 'polyflip.ai_lab.service.create_deployment_revision' imported but unused
- 1950:1: E302 expected 2 blank lines, found 1
- 70:1: F401 'polyflip.db.models.AIExperimentConfig' imported but unused
- *...и еще 4 ошибок*

### Файл: `polyflip/api/ai_lab_agent.py`
- 1011:1: E302 expected 2 blank lines, found 0

### Файл: `polyflip/api/analytics.py`
- 621:54: E231 missing whitespace after ','
- 622:38: E231 missing whitespace after ','
- 24:1: E402 module level import not at top of file
- 632:85: W291 trailing whitespace
- 694:75: E231 missing whitespace after ','
- *...и еще 131 ошибок*

### Файл: `polyflip/api/auth.py`
- 8:1: E302 expected 2 blank lines, found 1

### Файл: `polyflip/api/backtest_api.py`
- 42:1: E402 module level import not at top of file
- 28:1: E402 module level import not at top of file
- 24:1: E402 module level import not at top of file
- 13:1: E402 module level import not at top of file
- 22:1: E402 module level import not at top of file
- *...и еще 25 ошибок*

### Файл: `polyflip/api/backtest_schemas.py`
- 3:1: F401 'pydantic.field_validator' imported but unused
- 46:17: W291 trailing whitespace
- 153:1: E302 expected 2 blank lines, found 1
- 45:22: W291 trailing whitespace

### Файл: `polyflip/api/crypto_backtest_api.py`
- 71:9: F401 'polyflip.crypto.trainer.CRYPTO_FEATURES as DEFAULT_FEATURES' imported but unused
- 128:33: E221 multiple spaces before operator
- 127:34: E221 multiple spaces before operator
- 35:35: E221 multiple spaces before operator

### Файл: `polyflip/api/crypto_dashboard.py`
- 1641:28: W291 trailing whitespace
- 24:1: E402 module level import not at top of file
- 22:1: E402 module level import not at top of file
- 1645:75: W291 trailing whitespace
- 25:1: F401 'sqlalchemy.cast' imported but unused
- *...и еще 74 ошибок*

### Файл: `polyflip/api/dashboard.py`
- 28:1: E402 module level import not at top of file
- 14:1: F401 'sqlalchemy.and_' imported but unused
- 13:1: E402 module level import not at top of file
- 313:41: E712 comparison to True should be 'if cond is True:' or 'if cond:'
- 4:1: E402 module level import not at top of file
- *...и еще 21 ошибок*

### Файл: `polyflip/api/execution_api.py`
- 1878:1: W293 blank line contains whitespace
- 206:1: E402 module level import not at top of file
- 823:1: E402 module level import not at top of file
- 1664:1: W293 blank line contains whitespace
- 491:1: W293 blank line contains whitespace
- *...и еще 21 ошибок*

### Файл: `polyflip/api/main.py`
- 76:1: E305 expected 2 blank lines after class or function definition, found 1
- 79:1: E302 expected 2 blank lines, found 1
- 158:1: E302 expected 2 blank lines, found 1
- 96:1: W293 blank line contains whitespace
- 153:1: E302 expected 2 blank lines, found 1
- *...и еще 5 ошибок*

### Файл: `polyflip/api/mrf_api.py`
- 56:1: E305 expected 2 blank lines after class or function definition, found 1
- 246:45: E712 comparison to True should be 'if cond is True:' or 'if cond:'

### Файл: `polyflip/api/settings.py`
- 112:1: E402 module level import not at top of file
- 292:5: E303 too many blank lines (2)
- 13:1: F401 'polyflip.settings_registry.registry_defaults' imported but unused
- 112:1: E305 expected 2 blank lines after class or function definition, found 1
- 223:1: W293 blank line contains whitespace
- *...и еще 33 ошибок*

### Файл: `polyflip/api/slippage.py`
- 26:1: E302 expected 2 blank lines, found 1
- 9:1: E302 expected 2 blank lines, found 1

### Файл: `polyflip/api/trading_dashboard.py`
- 18:1: E402 module level import not at top of file
- 8:1: F401 'datetime.time as dt_time' imported but unused
- 9:1: E402 module level import not at top of file
- 7:1: E402 module level import not at top of file
- 255:1: E402 module level import not at top of file
- *...и еще 12 ошибок*

### Файл: `polyflip/backtesting/market_replay.py`
- 52:1: W293 blank line contains whitespace
- 10:1: F401 'polyflip.trading.feature_builder.signal_from_snapshot_row' imported but unused

### Файл: `polyflip/backtesting/metrics.py`
- 27:1: W293 blank line contains whitespace
- 81:1: W293 blank line contains whitespace
- 44:1: W293 blank line contains whitespace
- 59:1: W293 blank line contains whitespace

### Файл: `polyflip/backtesting/runner.py`
- 145:1: W293 blank line contains whitespace
- 141:1: W293 blank line contains whitespace
- 20:1: E302 expected 2 blank lines, found 1
- 37:1: W293 blank line contains whitespace
- 40:1: W293 blank line contains whitespace
- *...и еще 7 ошибок*

### Файл: `polyflip/backtesting/simulated_trader.py`
- 47:1: W293 blank line contains whitespace
- 43:1: W293 blank line contains whitespace
- 51:1: W293 blank line contains whitespace
- 53:1: W293 blank line contains whitespace

### Файл: `polyflip/collector/client.py`
- 275:1: W293 blank line contains whitespace
- 454:49: E261 at least two spaces before inline comment
- 457:1: W293 blank line contains whitespace
- 429:48: E261 at least two spaces before inline comment
- 301:1: W293 blank line contains whitespace
- *...и еще 22 ошибок*

### Файл: `polyflip/collector/parser.py`
- 111:37: E261 at least two spaces before inline comment
- 12:1: E302 expected 2 blank lines, found 1
- 49:25: E261 at least two spaces before inline comment
- 64:1: W293 blank line contains whitespace
- 129:1: W293 blank line contains whitespace
- *...и еще 2 ошибок*

### Файл: `polyflip/collector/resolver.py`
- 21:37: E701 multiple statements on one line (colon)
- 14:1: E302 expected 2 blank lines, found 1
- 89:29: E261 at least two spaces before inline comment
- 106:1: W293 blank line contains whitespace
- 35:13: F811 redefinition of unused 'json' from line 6
- *...и еще 26 ошибок*

### Файл: `polyflip/constants.py`
- 77:18: E221 multiple spaces before operator
- 78:18: E221 multiple spaces before operator
- 71:12: E221 multiple spaces before operator
- 76:17: E221 multiple spaces before operator
- 35:1: E305 expected 2 blank lines after class or function definition, found 1
- *...и еще 2 ошибок*

### Файл: `polyflip/crypto/backtester.py`
- 42:1: E402 module level import not at top of file
- 429:11: E221 multiple spaces before operator
- 202:5: E303 too many blank lines (2)
- 38:1: F401 'polyflip.crypto.trainer.CRYPTO_FEATURES' imported but unused
- 262:41: E272 multiple spaces before keyword
- *...и еще 24 ошибок*

### Файл: `polyflip/crypto/binance_client.py`
- 34:1: E302 expected 2 blank lines, found 1

### Файл: `polyflip/crypto/candle_collector.py`
- 88:1: W391 blank line at end of file
- 21:8: E221 multiple spaces before operator
- 14:1: F401 'polyflip.crypto.binance_client.COIN_TO_SYMBOL' imported but unused

### Файл: `polyflip/crypto/candle_repository.py`
- 92:1: W293 blank line contains whitespace

### Файл: `polyflip/crypto/dataset.py`
- 12:1: F401 'datetime.datetime' imported but unused
- 97:1: W293 blank line contains whitespace
- 82:1: W293 blank line contains whitespace
- 13:1: F401 'numpy as np' imported but unused
- 12:1: F401 'datetime.timezone' imported but unused
- *...и еще 9 ошибок*

### Файл: `polyflip/crypto/edge.py`
- 29:1: E302 expected 2 blank lines, found 1
- 5:1: E302 expected 2 blank lines, found 1

### Файл: `polyflip/crypto/experiment_configs.py`
- 103:1: E305 expected 2 blank lines after class or function definition, found 1

### Файл: `polyflip/crypto/feature_builder.py`
- 130:59: E221 multiple spaces before operator
- 170:8: E221 multiple spaces before operator
- 113:65: E272 multiple spaces before keyword
- 137:52: E221 multiple spaces before operator
- 175:9: E221 multiple spaces before operator
- *...и еще 67 ошибок*

### Файл: `polyflip/crypto/funding_collector.py`
- 31:1: W293 blank line contains whitespace
- 18:1: E302 expected 2 blank lines, found 1

### Файл: `polyflip/crypto/historical_loader.py`
- 109:1: W293 blank line contains whitespace
- 41:11: E221 multiple spaces before operator
- 120:1: W293 blank line contains whitespace
- 23:23: E221 multiple spaces before operator
- 42:13: E221 multiple spaces before operator
- *...и еще 3 ошибок*

### Файл: `polyflip/crypto/market_direction_service.py`
- 46:13: E131 continuation line unaligned for hanging indent

### Файл: `polyflip/crypto/market_outcome_dataset.py`
- 12:1: F401 'datetime.datetime' imported but unused
- 252:5: E303 too many blank lines (2)
- 12:1: F401 'datetime.timezone' imported but unused
- 67:1: E305 expected 2 blank lines after class or function definition, found 1

### Файл: `polyflip/crypto/market_regime.py`
- 17:1: F401 'pandas as pd' imported but unused

### Файл: `polyflip/crypto/market_regime_apply.py`
- 13:1: F401 'datetime.timezone' imported but unused
- 12:1: F401 'dataclasses.field' imported but unused
- 19:1: F401 'polyflip.crypto.market_regime_classifier.classify_global_regime' imported but unused
- 18:1: F401 'polyflip.crypto.market_regime.MIN_HISTORY_CANDLES' imported but unused
- 19:1: F401 'polyflip.crypto.market_regime_classifier.MarketPhase' imported but unused
- *...и еще 1 ошибок*

### Файл: `polyflip/crypto/market_regime_audit.py`
- 10:1: F401 'datetime.timezone' imported but unused
- 13:1: F401 'polyflip.crypto.market_regime.MIN_HISTORY_CANDLES' imported but unused
- 10:1: F401 'datetime.datetime' imported but unused

### Файл: `polyflip/crypto/market_regime_classifier.py`
- 12:1: F401 'typing.Sequence' imported but unused
- 14:1: F401 'polyflip.crypto.market_regime.BasketRegimeFeatures' imported but unused

### Файл: `polyflip/crypto/market_regime_integration.py`
- 26:1: F401 'polyflip.crypto.market_regime.AssetRegimeFeatures' imported but unused
- 25:1: F401 'polyflip.crypto.market_regime_classifier.classify_global_regime' imported but unused

### Файл: `polyflip/crypto/market_regime_policy.py`
- 12:1: F401 'typing.Optional' imported but unused

### Файл: `polyflip/crypto/polymarket_backtest.py`
- 879:27: E203 whitespace before ':'
- 860:1: E302 expected 2 blank lines, found 1
- 273:1: E303 too many blank lines (3)
- 229:1: E302 expected 2 blank lines, found 0
- 704:1: E302 expected 2 blank lines, found 1

### Файл: `polyflip/crypto/polymarket_join.py`
- 32:1: W293 blank line contains whitespace
- 14:1: F401 'sqlalchemy.text' imported but unused
- 12:1: F401 'datetime.timedelta' imported but unused

### Файл: `polyflip/crypto/predictor.py`
- 138:51: E261 at least two spaces before inline comment
- 66:5: E301 expected 1 blank line, found 0
- 287:17: E303 too many blank lines (2)
- 552:1: W293 blank line contains whitespace
- 550:1: W293 blank line contains whitespace
- *...и еще 17 ошибок*

### Файл: `polyflip/crypto/risk_guard.py`
- 47:15: E221 multiple spaces before operator
- 26:1: W293 blank line contains whitespace
- 9:23: E221 multiple spaces before operator
- 46:18: E221 multiple spaces before operator

### Файл: `polyflip/crypto/threshold_optimizer.py`
- 478:21: E701 multiple statements on one line (colon)
- 482:21: E701 multiple statements on one line (colon)

### Файл: `polyflip/crypto/trainer.py`
- 368:18: F841 local variable 'X_val' is assigned to but never used
- 1012:15: E221 multiple spaces before operator
- 1047:1: W293 blank line contains whitespace
- 33:1: F401 'polyflip.crypto.feature_builder.build_features' imported but unused
- 369:18: F841 local variable 'y_val' is assigned to but never used
- *...и еще 22 ошибок*

### Файл: `polyflip/db/connection.py`
- 8:1: E302 expected 2 blank lines, found 1

### Файл: `polyflip/db/init_runtime_settings.py`
- 121:1: E302 expected 2 blank lines, found 1
- 101:1: E302 expected 2 blank lines, found 1
- 15:1: E402 module level import not at top of file
- 88:1: W293 blank line contains whitespace

### Файл: `polyflip/db/models.py`
- 223:24: E221 multiple spaces before operator
- 1758:14: E221 multiple spaces before operator
- 1765:1: E302 expected 2 blank lines, found 1
- 226:22: E221 multiple spaces before operator
- 1702:1: E302 expected 2 blank lines, found 1
- *...и еще 45 ошибок*

### Файл: `polyflip/debug_backtest.py`
- 16:1: W293 blank line contains whitespace
- 10:1: E302 expected 2 blank lines, found 1
- 83:1: W293 blank line contains whitespace
- 2:1: F401 'datetime.datetime' imported but unused
- 65:1: W293 blank line contains whitespace
- *...и еще 5 ошибок*

### Файл: `polyflip/execution/contracts.py`
- 138:1: E303 too many blank lines (3)

### Файл: `polyflip/execution/order_strategies.py`
- 508:1: E302 expected 2 blank lines, found 1
- 54:1: E303 too many blank lines (3)

### Файл: `polyflip/execution/release_gate.py`
- 731:9: E303 too many blank lines (2)

### Файл: `polyflip/models/feature_lags.py`
- 27:1: E302 expected 2 blank lines, found 1
- 11:1: F401 'numpy as np' imported but unused
- 50:5: E303 too many blank lines (2)
- 30:1: W293 blank line contains whitespace

### Файл: `polyflip/models/sequence_features.py`
- 57:1: E305 expected 2 blank lines after class or function definition, found 1

### Файл: `polyflip/models/temporal_validation.py`
- 5:1: F401 'typing.Iterator' imported but unused

### Файл: `polyflip/models/trainer.py`
- 472:1: W293 blank line contains whitespace
- 110:26: E221 multiple spaces before operator
- 676:1: W293 blank line contains whitespace
- 666:73: W291 trailing whitespace
- 644:1: W293 blank line contains whitespace
- *...и еще 46 ошибок*

### Файл: `polyflip/scheduler/jobs.py`
- 748:1: W293 blank line contains whitespace
- 241:1: W293 blank line contains whitespace
- 39:1: E302 expected 2 blank lines, found 1
- 795:1: W293 blank line contains whitespace
- 265:1: W293 blank line contains whitespace
- *...и еще 55 ошибок*

### Файл: `polyflip/scripts/compare_target_models.py`
- 13:1: F401 'datetime.timezone' imported but unused
- 69:11: F541 f-string is missing placeholders
- 81:1: W293 blank line contains whitespace
- 136:1: W293 blank line contains whitespace
- 48:1: W293 blank line contains whitespace
- *...и еще 4 ошибок*

### Файл: `polyflip/scripts/report_24h.py`
- 35:19: W291 trailing whitespace
- 16:19: W291 trailing whitespace
- 12:1: E302 expected 2 blank lines, found 1
- 69:19: W291 trailing whitespace
- 7:1: E302 expected 2 blank lines, found 1
- *...и еще 1 ошибок*

### Файл: `polyflip/scripts/report_models.py`
- 162:18: E221 multiple spaces before operator
- 97:10: E221 multiple spaces before operator
- 97:9: E741 ambiguous variable name 'l'
- 165:18: E221 multiple spaces before operator
- 168:16: E221 multiple spaces before operator
- *...и еще 10 ошибок*

### Файл: `polyflip/services/settings_service.py`
- 34:13: F841 local variable 'val' is assigned to but never used
- 7:1: F401 'typing.Optional' imported but unused
- 106:1: W391 blank line at end of file

### Файл: `polyflip/settings_registry.py`
- 403:1: E305 expected 2 blank lines after class or function definition, found 1
- 448:1: E302 expected 2 blank lines, found 1

### Файл: `polyflip/trading/combined_voting.py`
- 378:1: W293 blank line contains whitespace
- 1862:1: W293 blank line contains whitespace
- 387:1: E302 expected 2 blank lines, found 1
- 1555:1: W293 blank line contains whitespace
- 375:1: W293 blank line contains whitespace
- *...и еще 11 ошибок*

### Файл: `polyflip/trading/decision_logic.py`
- 115:13: E128 continuation line under-indented for visual indent
- 44:1: E305 expected 2 blank lines after class or function definition, found 1
- 26:1: E302 expected 2 blank lines, found 1
- 67:1: E303 too many blank lines (3)
- 13:1: F401 'polyflip.trading.position_sizing.compute_bet_size_edge_scaled' imported but unused
- *...и еще 7 ошибок*

### Файл: `polyflip/trading/decision_runners.py`
- 697:1: W293 blank line contains whitespace
- 1:1: F401 'dataclasses' imported but unused
- 795:5: E303 too many blank lines (2)
- 1368:9: F841 local variable 'mrf_audit_dict' is assigned to but never used
- 189:1: W293 blank line contains whitespace
- *...и еще 21 ошибок*

### Файл: `polyflip/trading/engine.py`
- 6:1: F401 'typing.Optional' imported but unused
- 21:1: F401 'polyflip.crypto.candle_repository.get_recent_candles' imported but unused
- 9:1: F401 'polyflip.constants.TRADING_MODE_COMBINED' imported but unused

### Файл: `polyflip/trading/feature_builder.py`
- 46:31: E116 unexpected indentation (comment)
- 33:1: E302 expected 2 blank lines, found 1
- 46:31: E114 indentation is not a multiple of 4 (comment)

### Файл: `polyflip/trading/funnel_logger.py`
- 39:37: E221 multiple spaces before operator
- 45:37: E221 multiple spaces before operator
- 41:31: E221 multiple spaces before operator
- 43:35: E221 multiple spaces before operator
- 38:36: E221 multiple spaces before operator
- *...и еще 2 ошибок*

### Файл: `polyflip/trading/market_guards.py`
- 52:1: W293 blank line contains whitespace
- 47:1: W293 blank line contains whitespace
- 1:1: F401 'dataclasses' imported but unused
- 65:1: W293 blank line contains whitespace

### Файл: `polyflip/trading/market_loader.py`
- 39:1: W293 blank line contains whitespace
- 23:1: W293 blank line contains whitespace
- 56:1: W293 blank line contains whitespace
- 26:1: W293 blank line contains whitespace
- 35:1: W293 blank line contains whitespace

### Файл: `polyflip/trading/ml_inference.py`
- 36:1: W293 blank line contains whitespace
- 20:57: E261 at least two spaces before inline comment
- 30:1: E302 expected 2 blank lines, found 1
- 225:1: W293 blank line contains whitespace
- 68:1: W293 blank line contains whitespace
- *...и еще 18 ошибок*

### Файл: `polyflip/trading/position_sizing.py`
- 69:1: E302 expected 2 blank lines, found 1
- 48:1: W293 blank line contains whitespace
- 11:1: E302 expected 2 blank lines, found 1
- 51:1: W293 blank line contains whitespace

### Файл: `polyflip/trading/pre_trade_validator.py`
- 139:1: W293 blank line contains whitespace
- 278:5: E303 too many blank lines (3)
- 275:1: W293 blank line contains whitespace
- 235:1: W293 blank line contains whitespace
- 1:1: F401 'dataclasses' imported but unused
- *...и еще 8 ошибок*

### Файл: `polyflip/trading/schemas.py`
- 34:1: E302 expected 2 blank lines, found 1
- 69:1: E302 expected 2 blank lines, found 1
- 18:1: E302 expected 2 blank lines, found 1
- 60:1: E305 expected 2 blank lines after class or function definition, found 1

### Файл: `polyflip/trading/settings_loader.py`
- 4:1: F401 'polyflip.config.settings' imported but unused
- 18:1: W293 blank line contains whitespace

### Файл: `polyflip/trading/stoploss.py`
- 3:1: F401 'typing.Optional' imported but unused

### Файл: `polyflip/trading/stoploss_worker.py`
- 2:1: F401 'asyncio' imported but unused
- 3:1: F401 'datetime.timedelta' imported but unused
- 8:1: F401 'polyflip.db.models.SlippageLog' imported but unused
- 84:1: W293 blank line contains whitespace
- 86:1: W293 blank line contains whitespace

### Файл: `polyflip/trading/takeprofit_worker.py`
- 139:1: W293 blank line contains whitespace
- 30:5: F841 local variable 'fee_rate' is assigned to but never used
- 141:1: W293 blank line contains whitespace
- 156:1: W293 blank line contains whitespace
- 7:1: F401 'polyflip.db.models.SlippageLog' imported but unused

### Файл: `polyflip/trading/trade_recorder.py`
- 106:5: F841 local variable 'strk_src' is assigned to but never used
- 115:66: W291 trailing whitespace
- 1:1: F401 'dataclasses' imported but unused
- 327:25: E261 at least two spaces before inline comment
- 91:5: F841 local variable 'dir_prob' is assigned to but never used
- *...и еще 35 ошибок*

### Файл: `polyflip/trading/trading_config.py`
- 172:1: E302 expected 2 blank lines, found 1
- 19:1: E302 expected 2 blank lines, found 1
- 1:1: F401 'dataclasses' imported but unused
- 162:1: W293 blank line contains whitespace
- 169:1: W293 blank line contains whitespace

### Файл: `polyflip/trading/utils.py`
- 10:1: W293 blank line contains whitespace
- 13:1: W293 blank line contains whitespace
- 11:72: W291 trailing whitespace

### Файл: `polyflip/trading/weighted_benchmark.py`
- 102:1: E302 expected 2 blank lines, found 1
- 551:1: E302 expected 2 blank lines, found 1
- 1587:1: E303 too many blank lines (4)
- 583:1: E302 expected 2 blank lines, found 1

### Файл: `polyflip/trading/weighted_policy.py`
- 835:1: E303 too many blank lines (3)

### Файл: `polyflip/trading/weighted_sizing.py`
- 141:1: E303 too many blank lines (3)


## 3. Результаты тестов (Pytest)
✅ Все тесты проходят успешно.

**Предупреждения (Warnings) при прогоне тестов (27 уникальных):**
-   /app/tests/test_engine_crypto_sizing.py:57: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
-   /app/tests/crypto/test_market_outcome_dataset.py:31: DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated, and will raise an error in the future. This includes implicit conversion of bare integers (e.g. `+ 1`).Please use a specific unit instead.
-   /app/tests/crypto/test_dataset_alignment.py:20: DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated, and will raise an error in the future. This includes implicit conversion of bare integers (e.g. `+ 1`).Please use a specific unit instead.
-   /app/tests/crypto/test_lgbm_def_experiment.py:28: DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated, and will raise an error in the future. This includes implicit conversion of bare integers (e.g. `+ 1`).Please use a specific unit instead.
-   /home/jules/.cache/pypoetry/virtualenvs/polyflip-9TtSrW0h-py3.12/lib/python3.12/site-packages/_pytest/fixtures.py:1313: PytestRemovedIn10Warning: Class-scoped fixture defined as instance method is deprecated.
-   /app/tests/crypto/test_market_outcome_dataset.py:32: DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated, and will raise an error in the future. This includes implicit conversion of bare integers (e.g. `+ 1`).Please use a specific unit instead.
-   /app/polyflip/crypto/market_outcome_dataset.py:287: DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated, and will raise an error in the future. This includes implicit conversion of bare integers (e.g. `+ 1`).Please use a specific unit instead.
-   /home/jules/.cache/pypoetry/virtualenvs/polyflip-9TtSrW0h-py3.12/lib/python3.12/site-packages/pydantic/main.py:263: UserWarning: favorite_threshold=0.1 < 0.5. PURE_FAVORITE стратегия будет торговать аутсайдеров. Используйте OUTSIDER стратегию вместо этого.
-   /app/polyflip/crypto/market_outcome_dataset.py:265: DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated, and will raise an error in the future. This includes implicit conversion of bare integers (e.g. `+ 1`).Please use a specific unit instead.
-   /app/polyflip/crypto/polymarket_backtest.py:254: DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated, and will raise an error in the future. This includes implicit conversion of bare integers (e.g. `+ 1`).Please use a specific unit instead.
- *...и еще 17 предупреждений*
