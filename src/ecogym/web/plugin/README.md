# @ecogym/dsh-plugin

ecogym 的 DeepSeek Harness 插件：DSH 启动时启动 ecogym 服务，DSH 侧栏多一个「生态进化」面板。安装与使用见仓库 README 的“在 DSH 中使用”；本文说明插件和 DSH 之间的约定。已在 DSH 0.1.5-rc.2 上验证。

## 组成

| 文件 | 作用 |
|---|---|
| `package.json` | `dsh.bundle.patch` 让 `dsh plugin add` 把本包加为 profile 的一层；`dsh.client` 声明浏览器端 `lib/client.js` |
| `cordis.patch.yml` | 这一层的内容：插入 id 为 `ecogym` 的插件行 |
| `lib/index.js` | 宿主端（DSH 进程内）：启动、看护、停止 ecogym 服务；在 `/api/ecogym/*` 提供同源代理 |
| `lib/client.js` | 浏览器端：`sidebar.panellist` 里的面板入口，`main` 槽位（键 `ecogym`）里的面板 |

## 宿主端

- 设置来自插件行的 `config`（含 `python` 时），否则来自 `~/.ecogym/dsh-plugin.json`（`$ECOGYM_DSH_SETTINGS`），每次启动服务时重新读取：`python`、`home`（ECOGYM_HOME）、`ledger`、`env`。
- 启动 `<python> -m ecogym serve --host 127.0.0.1 --port 0 --exit-with-stdin`，从输出里读端口；令牌随机生成，经环境变量 `ECOGYM_SERVICE_TOKEN` 交给服务，不出 DSH 进程。`PATH` 前面加上运行 DSH 的 node 所在目录，智能体能找到 `dsh`。
- 服务就绪前退出视为启动失败，连续 3 次后停止重试，等面板里的「重新启动后端」；就绪后崩溃按 2、5、15、30、60 秒退避重启。重启时等旧服务退出后才启动新服务，重复的重启请求合并为一次。
- DSH 卸载插件或退出时关闭服务的标准输入并发 SIGTERM，40 秒后 SIGKILL。DSH 自己最多等 5 秒就退出，所以清理由服务独立完成：最多 30 秒内停下智能体和裁判（进化运行记为已停止，可继续）；到时仍没结束的步骤，服务把它的运行记为已停止后直接退出。SIGINT（连按 Ctrl-C）和 SIGHUP（关闭终端）按同样方式处理。
- 进化运行的每一代在账本里记下运行它的进程（pid 与主机名）。账本显示“运行中”但这个进程已不在时，面板显示“已中断”，可以直接继续；进程还在时显示“运行中（其他进程）”，继续需要确认。

## 代理

- 只接受 GET 和 POST。请求先过 DSH 的 `connection.requestRejection`（与 DSH 自己的 `/api` 相同：Host 必须是回环或 `--trusted-host`，跨站请求和不同源的 Origin 被拒，浏览器须带登录 cookie）；DSH 没有这个服务时返回 503，接口不开放。
- 转发时去掉 `Cookie`、`Authorization`、`Origin`、`Referer` 和逐跳头，换上服务令牌；响应去掉 `Set-Cookie`。POST 请求体整个读入后带 `Content-Length` 转发，上限 2 MB（超过返回 413）；超时 120 秒；浏览器断开时对服务的请求一起断开。
- `GET /api/ecogym/_status` 返回服务状态与最近的输出；`POST /api/ecogym/_restart`（需 `X-Ecogym: 1`）重启服务。其余路径转发到服务的 `/api/*`。

## 浏览器端

- 只注入 `slots`，其他 DSH 服务按需读取；插件出错不会让 DSH 启动失败，面板内部的错误也只影响面板。
- 模型下拉列表来自服务的 `/api/catalog` 的 `models`：服务读 DSH 设置里的提供方，给出每个提供方能否运行、密钥是否存在（不含密钥本身）。DeepSeek 官方接口的模型清单另从 DSH 的 `session/modelCatalog` 读取，读不到就用 ecogym 的默认清单。
- 只用 DSH 平台模块（React 与 `@deepseek-ai/dsh-client-ui-primitives`：`Button`、`Tag`、`StateDot`、`Modal`、`MarkdownText`），某个组件不存在时退回到同样外观的本地实现；颜色和字号全部取 DSH 的设计令牌，跟随深浅色主题。
- 面板调用的全部接口集中在 `client.js` 的 `Api` 表；服务一侧的接口表是 `service.py` 的 `ROUTES`（`GET /api/ecogym/routes`）。分层、请求路径和每个页面用哪些接口见仓库的 `docs/architecture.md`。
- 以链接方式安装时，修改 `lib/client.js` 后刷新页面即可（2026-10-30 在 DSH 0.1.5-rc.2 上确认）；修改 `lib/index.js` 要重启 DSH。
