# 历史预设

更新：2026-10-17。[historical_presets/](historical_presets/) 保存已退出顶层预设列表的 JSON。当前预设与实验计划见 [configs/README.md](../../configs/README.md)。

包含 G13/G15/G17、B15、B30–B38、中间 B41/B42/B43/B44/B46、V40 和早期 A2/V5 等分支。B40、B47、B48、G20 也在 10 月审计中退出当前推荐：此前的短训练排名不能代表充分训练后的表现，B45 相关门控还存在退化问题。详细对照见[E1–E22 实验记录](../../reading/BREAKTHROUGH_E1_E22.md)。

可从仓库根目录通过显式路径检查旧配置展开后的命令：

```bash
.venv/bin/python tools/run_preset.py --preset archive/configs/historical_presets/g15_echf_main.json --dataset cora --dry_run
```

当前 runner 能加载文件并不等于恢复了当时的代码、默认值、训练预算和评价协议。严格追溯还需匹配历史工作副本及数据版本；不要把旧配置名中的 `main` 或归档说明中的排名当作当前推荐。
