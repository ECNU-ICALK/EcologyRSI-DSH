# 工程示例

这些示例用于验证 0.9.0 的 API、编排与账本重放，不需要模型 API key，也不代表真实作物预测性能。真实数据、Harness 和模型配置见[仓库 README](../README.md)。

## 确定性演示

`minimal_run.py` 使用确定性 DSH 替身和合成评测器，写入 SQLite 后关闭连接，再创建新的 director 重放事件。`task-manifest.json` 是该示例的冻结输入：数据 ID 与摘要绑定 seed-0 工程数据，任务 seed 仅控制候选搜索。

从仓库根目录执行，数据库请使用新的临时路径：

```bash
source .venv/bin/activate
PYTHONPATH=src python examples/minimal_run.py --db /tmp/ecologyrsi-example.sqlite3
```

安装 wheel 后无需设置 `PYTHONPATH`：

```bash
ecologyrsi-dsh demo --db /tmp/ecologyrsi-demo.sqlite3
ecologyrsi-dsh doctor --db /tmp/ecologyrsi-demo.sqlite3
```

## 单独调试 Host

`local-config.json` 提供示例数据库、回环地址、8765 端口和任务清单的相对路径配置。它不启动 Harness，也不自动使真实模型就绪：

```bash
source .venv/bin/activate
PYTHONPATH=src python -m ecologyrsi_dsh serve --config examples/local-config.json
```

`dsh-native-inference.md` 是原生 provider 配置示例；请使用自己的服务端环境变量或私有凭据文件，不把密钥写入示例或提交 Git。
