# 0.11.0 交付验收

实际验收结果记在 [轻量化记录](docs/LIGHTWEIGHT.md)。下列是交付要求，不能用历史版本结果替代。

- 版本、preset、源码与 wheel/sdist/native 插件内容一致；归档无凭据、原始数据、账本或私有会话。
- 仅接受当前 native joint@2 合同和新版本数据库；当前版本的检查点、暂停/恢复/取消、幂等和重启回放有效。
- 四角色真实预检独立绑定；同模型配置可用，缺少角色或凭据不得运行。
- 200 基础 + 100 条件预算、85 唯一起点、拟合/校准/选择/验证/最终测试隔离不变。
- 预测先封存再评分；九格覆盖、父子同组比较、跨日证据和非退化门禁保持；故障不能伪装为科学成功。
- 网页创建、运行管理、过程、候选与独立验证可用；网络和模型错误明确显示。
- 完整 Python/Node 测试、前端 smoke、lint、锁文件与固定 Harness 角色验收通过；记录实际数量和跳过项。
- 干净环境安装 wheel，检查 CLI doctor、健康接口、模型目录和静态资源；核对交付包全部摘要。
- 真实远程模型进化与科学收益单独记录，不能由确定性替身测试推定。

```bash
export ECOLOGYRSI_DATA_ROOT=/absolute/path/to/greenhouse
export DSH_BIN=/absolute/path/to/dsh
make verify
make lint
python scripts/source_metrics.py
# 提交最终源码后，从干净工作树构建。
make release
```
