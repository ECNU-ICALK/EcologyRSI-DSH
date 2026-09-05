# Task 1：公共合同与错误模型

## 改动

- 新增 `api/errors.py`：定义 `ErrorCode`，提供统一 `error_payload` 与异常到稳定错误码的安全映射。
- 新增 `api/contracts.py`：提供脱敏的 `build_command_receipt` 收据序列化器。
- `command_receipts.py`、执行轮询、HTTP transport、handler/DSH sidecar 错误路径统一接入合同 builder/mapper。
- 保留既有 HTTP 状态码、错误摘要脱敏和 DSH 未获准错误码不外泄的兼容语义。
- 增加合同测试，覆盖稳定 code、retryable、command_id、收据脱敏及 draining timeout 专用 code。

## 测试

- `PYTHONPATH=src .venv/bin/python -m unittest tests.test_api_contracts tests.test_command_receipts tests.test_public_redaction tests.test_http`：44 tests，OK。
- `PYTHONPATH=src .venv/bin/python -m unittest tests.test_dsh_tool_contracts tests.test_dsh_native_runtime`：80 tests，OK。

## 未解决风险

- 旧 DSH sidecar 合同对未获准的 provider 错误码仍省略 `error_code`，以保持现有兼容测试与安全边界；后续若协议版本允许，可统一为显式 `internal_error`。
- 其他阶段的 projection、ledger 热路径和浏览器协议尚未改动。
