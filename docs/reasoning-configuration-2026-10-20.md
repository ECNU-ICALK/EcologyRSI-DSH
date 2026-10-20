# 模型配置与推理能力

适用版本：EcologyRSI-DSH 0.11.0，Harness 0.2.0-rc.2。

## 保留可重启的配置

Python Host 读取 `llm-pi-ai.providers` 格式的私有目录文件；Harness 接收相同内容生成的 Cordis patch。将这两个文件放在 `DSH_HOME` 外的稳定目录，避免 Harness 首次启动导入并重命名其根目录 `settings.yaml` 后，Host 重启丢失模型列表。

从已有且已核对的配置生成一对文件，不改变模型、路由、凭据引用或推理能力：

```bash
node scripts/configure_dsh_reasoning.mjs \
  --dsh-package /absolute/path/to/node_modules/@deepseek-ai/dsh/package.json \
  --source /absolute/path/to/current-model-settings.yaml \
  --output "$PWD/.runtime/model-config" \
  --copy-only

export ECOLOGYRSI_DSH_SETTINGS_FILE="$PWD/.runtime/model-config/settings.yaml"
export ECOLOGYRSI_DSH_CREDENTIALS_FILE="$DSH_HOME/.credentials.yaml"
dsh web --patch "$PWD/.runtime/model-config/settings.patch.yml" \
  --no-open --host 127.0.0.1 --port 8848
```

`--source` 必须是实际已有的模型配置，至少包含 `llm-pi-ai.providers`。`--dsh-package` 指向已安装 Harness 的 package.json；不要填本项目原生插件的 package.json。配置生成器创建权限为 0600 的两个文件，拒绝覆盖已有输出；新配置请写入新目录并一起切换 Host 和 Harness。凭据由既有环境变量或专用凭据文件提供，不写入命令输出。

启动 Python sidecar 的完整步骤见 [README](../README.md)。Host、Harness 的令牌必须一致；模型配置、能力或推理策略改变后须新建运行并重新预检，不能改写旧运行的冻结身份。

## 显式调整推理能力

`--copy-only` 适用于保留已有配置，包括 `deepseek-v4.1-flash-1017`。若要为生成器已经支持的模型声明推理能力，用一个或多个 `--route provider/model` 替代 `--copy-only`。两种模式不能混用。

生成器目前支持 GLM-5.2、GLM-5.3/Flash、GLM-5/5.1/4.7，以及其正则明确列出的 DeepSeek V4 Flash/Pro 名称。它会拒绝未知名称、冲突的能力声明和有歧义的 provider 默认值；不会由相似型号推断 v4.1 的 wire 协议。具体白名单由 [生成器](../scripts/configure_dsh_reasoning.mjs) 定义。

模型可选 effort 与 provider 默认值是不同配置。原生样本角色只在模型明确声明支持 off 时关闭可选推理；轮次评审在支持 low 时使用 low。其他情况沿用声明的默认值或角色解析结果。策略与评审可以使用同一模型，但必须分别通过研究、轮次评审、样本规划、样本评审四个工具合同预检。

这些配置只声明请求语义。第三方网关的实际支持、授权和质量由真实预检与实验确认；HTTP 403 不等于该模型本身不具备工具能力。

## 验证

```bash
DSH_BIN=/absolute/path/to/dsh node \
  integrations/dsh_ecology_plugin/test/reasoning_registration_harness.mjs \
  /absolute/path/to/config-registration-proof.json
```

该回归使用临时配置、真实 Cordis/Harness 与本地模拟 HTTP，覆盖配置生成、私有文件权限、拒绝覆盖、模型注册、默认推理档位、出站请求字段及 copy-only 配置等价。它不发送远程模型请求，不证明预测质量或科学收益。换 Harness 版本后应重新执行。
