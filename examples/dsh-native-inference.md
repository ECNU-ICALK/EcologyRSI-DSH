# 为逐时点 Agent 配置独立的推理档位

不要只按模型名称判断实际推理行为。一个逐时点任务包含完整目标向量、可选模型工具、结构化提交以及可能的独立评审，较深的默认推理可能耗尽单次输出预算。

本地测试使用 GLM 5.2 的独立配置。把下列 provider 项合并到自己的 DSH `settings.yaml`，保留已有 provider；凭据通过 `GLM_API_KEY` 环境变量或 DSH 凭据文件提供，文件权限为 `0600`：

```yaml
llm-pi-ai:
  providers:
    glm-inference:
      displayName: GLM 逐时点推理
      apiKeyEnv: GLM_API_KEY
      api: openai-completions
      baseURL: https://api.pjlab.org.cn/v1
      reasoning: "off"
      compat:
        thinkingFormat: zai
        supportsReasoningEffort: true
        maxTokensField: max_tokens
      models:
        - id: glm-5.2
          reasoningEfforts:
            off: null
            minimal: minimal
            high: high
            max: max
```

这个组合让 DSH 实际发送 `thinking: {type: disabled}`。只给一个未声明推理能力的自定义模型设置 off，并不能保证关闭供应商默认推理。其他模型和供应商应核查自己的参数支持情况；不要把此配置推广到所有模型。

重启 DSH 和后端，在网页选择该策略模型以及独立评审模型，重新执行真实预检并创建新运行。模型配置摘要随运行冻结，同名配置改变不能无声混入已有实验。运行原始配置和秘密凭据不需要进入交付包。

GLM 5.2 参数依据：[官方推理说明](https://docs.z.ai/guides/capabilities/thinking)。本项目在 Pjlab 路由做了实际调用验证；相同参数在其他代理上的效果仍需验证。

独立评审也需要明确的推理档位。Kimi K3 始终推理，官方默认 `reasoning_effort=max`；本地逐起点评审改用 `low`，并为新运行冻结 4096 Token 的评审输出上限。与 GLM 的关闭推理方式不同，不应给 Kimi K3 套用 `thinking: disabled`。[Kimi K3 官方用法](https://github.com/MoonshotAI/Kimi-K3#6-model-usage)

对应 DSH provider 示例（合并到 `llm-pi-ai.providers`）：

```yaml
kimi-review:
  displayName: Kimi 有界评审
  apiKeyEnv: KIMI_API_KEY
  api: openai-completions
  baseURL: https://api.pjlab.org.cn/v1
  reasoning: low
  compat:
    supportsReasoningEffort: true
    maxTokensField: max_tokens
  models:
    - id: kimi-k3
      reasoningEfforts:
        low: low
        high: high
        max: max
```

配置别名共享同一供应商账户时，不会增加账户请求额度。默认六秒子任务启动间隔仍需结合实际限流反馈退避。
