# EcologyRSI-DSH

面向农业生态时序预测的循环自进化工作台。模型根据研究资料和实验反馈提出修改，Python Host 在冻结数据、预算和评分规则下验证。当前任务是温室气温、相对湿度和 CO₂ 的 1／6／24 小时预测。

**0.11.0 轻量版**只维护一条产品路径：网页 → DeepSeek Harness 原生角色 → 联合进化 `joint_plugin_epoch@2`。Python 3.10+，DeepSeek Harness **0.2.0-rc.2**，建议 Node.js 24 LTS。Python 包无第三方运行依赖。

这是离线预测研究。执行成功、搜索采用与独立验证是三个不同结论；运行可以正常完成而没有获得合格改进。模型不能修改评分规则、读取未授权分区、执行任意生成代码或控制真实设备。

## 安装与启动

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
npm install --global @deepseek-ai/dsh@0.2.0-rc.2
export DSH_HOME="$PWD/.runtime/dsh-home"
ecologyrsi-dsh install-dsh-runtime --profile web
```

在专用 Harness 中配置 provider、模型和凭据。将 Host 的目录文件保存在 `.runtime/model-config/settings.yaml`，并从同一配置生成 Harness 使用的 `settings.patch.yml`，两者均仅当前用户可读。生成方法见[配置说明](docs/reasoning-configuration-2026-10-20.md)。密钥不进入网页或 Git。策略和评审可以选择同一个具备双角色权限的模型；例如在已有 provider 中追加 `deepseek-v4.1-flash-1017`，目录 ID 为 `pjlab/deepseek-v4.1-flash-1017`。路由是否可用由实际预检确认。

```bash
export ECOLOGYRSI_DATA_ROOT="$HOME/.ecologyrsi-dsh/data/greenhouse"
ecologyrsi-dsh data fetch agc_cucumber_2018
# 需要番茄时再下载；解压需要 bsdtar。
# ecologyrsi-dsh data fetch agc_tomato_2019
ecologyrsi-dsh data audit

export ECOLOGYRSI_DSH_SETTINGS_FILE="$PWD/.runtime/model-config/settings.yaml"
export ECOLOGYRSI_DSH_CREDENTIALS_FILE="$DSH_HOME/.credentials.yaml"
export ECOLOGYRSI_DSH_RUNTIME_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export ECOLOGYRSI_SIDECAR_TOOL_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export ECOLOGYRSI_DSH_RUNTIME_URL="http://127.0.0.1:8848"
mkdir -p .runtime

ecologyrsi-dsh serve --host 127.0.0.1 --port 8777 \
  --db "$PWD/.runtime/lightweight-0.11.sqlite3" &
ECOLOGYRSI_SIDECAR_PID=$!
trap 'kill "$ECOLOGYRSI_SIDECAR_PID" 2>/dev/null || true' EXIT
dsh web --patch "$PWD/.runtime/model-config/settings.patch.yml" \
  --no-open --host 127.0.0.1 --port 8848
```

打开 [工作台](http://127.0.0.1:8848/plugins/ecology/evolution/?api=/api/ecology-evolution)。选择就绪数据集、episode 和两个角色模型，核对轮数与容量，点击创建。系统分别验证研究、轮次评审、样本规划和样本评审四个角色的真实工具与结构化合同；同模型也必须独立通过四份预检。

Host 的目录文件与 Harness 的 patch 必须来自同一份配置。不要把 Host 的持久输入指向 `DSH_HOME/settings.yaml`：Harness 首次导入可能将它改名为 `settings.yaml.imported`，随后 Host 重启会失去目录。配置调整后共同重启 Host 与 Harness，令牌也须一致。

## 联合进化流程

每轮四个候选可组合修改提示词、已登记模型参数、特征、Skill、受控数值程序与经验记忆。筛选后只深入一条主线，再与轮初最佳版本比较。

| 阶段 | 对照设计 | 最大执行次数 |
|---|---|---:|
| 同组初筛 | 四候选与轮初最佳，使用相同 10 个起点 | 50 |
| 局部配对 | 父版本与一次局部修订，使用 25 个新起点 | 50 |
| 轮末比较 | 候选与轮初最佳，使用 50 个新起点 | 100 |
| 条件复核 | 满足基础规则但证据临界时，双臂复测一次 | 100 |
| 合计 | 基础 200，加最多 100 复核预留 | **300** |

每轮需要 **85 个不同评分起点**，还需历史上下文与目标成熟隔离。一个起点有三个目标 × 三个时距，共九个评分单元。执行次数不等于 LLM 请求数，也不等于独立观测数。容量不足时减少轮数；不缩短隔离或复用留出数据。

采用新版需同时满足执行完整性、实用增益、分项不退化和跨日证据。条件复测要求首次和复测均通过，不能挑较好的一次。模型评审提供建议，Host 决定是否满足冻结门禁。

## 数据与证据

真实支持 AGC 2018 黄瓜、AGC 2019 番茄数据；来源与许可见 [datasets](datasets)。数据文件不随源码分发。文件就绪和来源归档校验分开显示。

数据按时间划为拟合、不确定性校准、搜索选择、独立验证与最终测试，并保留隔离段。搜索只能使用拟合和反馈区；独立验证不回流调参。科学比较固定父子版本、同组时间点、评分规则与预算。当前不支持跨 episode 联合认证。

在工作台查看运行进度、候选修改、九格预测结果与配对增益。不同批次原始分数不能当作修改收益曲线。暂停后等待在途任务收尾；恢复沿用当前版本已冻结的合同与检查点。

## 轻量版边界

0.11 不读取或迁移旧运行库。请使用新数据库，历史实验使用 [0.10.11 基线](https://github.com/ECNU-ICALK/EcologyRSI-DSH/tree/codex/delivery-0.10.11) 与原运行目录读取。

删除的产品路径包括旧直连模型网关、旧实验协议、合成演示、专家咨询、人工文字调参与训练资产导出。保留当前版本的暂停/恢复/取消、归档/删除、数据预览、科学轨迹、独立验证，以及同模型的角色隔离与权限检查。具体删减、计量方法与验收见 [轻量化记录](docs/LIGHTWEIGHT.md)。

## 开发与验证

```bash
make test
make test-integration
make lint
python scripts/source_metrics.py
# 固定 Harness 可执行文件用于完整本地角色验收：
DSH_BIN=/absolute/path/to/dsh make verify
make release
```

`source_metrics.py` 对比 GitHub 基线提交与当前工作树，分别报告物理行、非空行和字节；测试、文档、JSON 与打包产物不计入运行代码压缩比例。工程验证使用确定性模型替身，不能据此宣称真实模型带来科学性能提升。

本项目采用专有许可，详见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
