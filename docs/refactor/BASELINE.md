# EcologyRSI-DSH 重构基线

> 历史记录：其中的独立实验内核和命令已在 0.4.0 移除，不代表当前架构或可用入口。当前实现见 [代码清理说明](CODE-CLEANUP-0.4.0.md)。

记录时间：2026-09-06（Asia/Shanghai）

## 工作区

- Git HEAD：`af265fc198518b5a17258f91cfec1eee8f27bbe9`
- 分支：`main`（与 `origin/main` 同步）
- 初始未提交改动（保留，不覆盖）：
  - `plugins/ecology_evolution/assets/js/commands.js`
  - `plugins/ecology_evolution/assets/js/render_shell.js`
  - `plugins/ecology_evolution/test/smoke.mjs`
  - `src/ecologyrsi_dsh/api/execution.py`
  - `tests/test_http.py`

## 运行时

- 系统 `python`：Python 2.7.13（不用于本项目）
- 项目解释器：`.venv/bin/python`，Python 3.12.13
- SQLite：3.50.4
- Node：v24.19.0
- DSH CLI：`0.1.0-rc.6`
- 项目要求：Python >=3.10，运行时依赖为空

## 原仓库离线基线

| 命令 | 结果 |
|---|---|
| `make test-fast` | PASS，81 tests，0 failures |
| `make compile` | PASS，exit 0 |
| `make test` | 已启动完整 `unittest discover`；基线记录过程中持续执行，完整结果见 `/tmp/ecology-legacy-test.txt`，不得将未读完输出标为通过 |

完整测试可能包含构建/安装与 Node 交互测试，涉及较长运行时间；本次不调用外部模型服务、远程 DSH 或凭据。

## 参考重构包基线

`EcologyRSI-Rebuild-Kit.zip` 在独立临时目录中使用项目 Python 复现：

- `python scripts/verify.py`：PASS，71 tests，0 failures
- `python scripts/ecology.py demo --out runs/codex-baseline --seed 7 --budget 4`：4 次评测，合成来源，baseline loss `0.5145338422391471`，champion loss `0.02595471482756949`，1 次晋级，`release_eligible=false`
- `python scripts/ecology.py verify --out runs/codex-baseline`：PASS，22 events，27 artifacts，audit `consistent`

上述数值只证明参考包的离线软件链路，不证明真实生态泛化。

## 关键旧入口

- Python 包：`ecologyrsi_dsh`
- 主 CLI：`ecologyrsi-dsh` → `ecologyrsi_dsh.application.cli:main`
- AI 实验 CLI：`ecologyrsi-ai-evolve` → `ecologyrsi_dsh.evolution_lab.__main__:main`
- Web/API 入口：`ecologyrsi_dsh.api`
- 原生 DSH 适配：`ecologyrsi_dsh.integrations`
- 旧测试入口：`make test-fast`、`make test`、`make test-integration`

## 数据与外部边界

仓库内包含示例温室数据与本地 SQLite 运行文件；未授权删除、迁移或覆盖。真实 DSH、外部模型、付费 API、正式确认数据和发布权限不作为默认基线依赖。
