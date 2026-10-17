# 当前配置与实验入口

更新：2026-10-17。所有命令从仓库根目录执行；环境安装见 [UV_ENVIRONMENTS.txt](../UV_ENVIRONMENTS.txt)。

## 模型预设

| `presets/` 文件 | 模型与用途 |
|---|---|
| [baseline_v1.json](presets/baseline_v1.json) | V1，edge-free DSE/LSENet；边增强实验必须保留的对照 |
| [b31_dualscalar_assign.json](presets/b31_dualscalar_assign.json) | V31，消息/SE 双标量权重和分配残差；`run_preset.py` 的默认预设 |
| [b45_v31_msgcond_gs050.json](presets/b45_v31_msgcond_gs050.json) | B31 加叶层消息门控，scale 0.50；原始重归一化门控在 Photo/AIFB 上崩塌，用于诊断对照 |

```bash
.venv/bin/python tools/run_preset.py --list_presets
.venv/bin/python tools/run_preset.py --preset baseline_v1 --dataset cora --dry_run
.venv/bin/python tools/run_preset.py --preset b31_dualscalar_assign --dataset cora --edge_ctx_dim 32 --dry_run
```

移除最后一条命令的 `--dry_run` 才会训练。预设主要描述边模型参数；便捷运行器提供 2000 epochs、
hidden size 512、weight decay 0、temperature 1.0、3 次解码、无提前停止、默认冷启动的训练预算。
`main.py` 的默认值不同；批量实验以各 plan 的 `defaults` / `overrides` 为准。

`--edge_ctx_dim` 向边属性追加标准化的对称端点乘积。`max_nums` 是层次预算，不保证最终簇数。
单次预设运行不比较模型族，也不进行 warm/cold 双启动选择。

## E1–E22 实验计划

[breakthrough/README.md](breakthrough/README.md) 提供计划清单、依赖顺序、历史完成状态、E14–E22 复现命令和报告口径。
其中 JSON 分为训练 plan 和候选选择 spec；CSV `e21_few_label_edges_jobs.csv` 是 CPU 工具输入清单，不能交给训练运行器。

```bash
# 先检查 E21 的 88 个展开任务；不会训练或写入结果
.venv/bin/python tools/run_breakthrough.py \
  --plan configs/breakthrough/e21_few_label_dse.json --root_path data --dry_run
```

批量运行器按“preset → plan defaults → job overrides”的优先级合并参数，展开 dataset × seed，
结果写入 `results/breakthrough_v1/<plan-name>/`。便捷 CLI 未暴露的选项应写入 plan，例如 E9 的
`edge_weight_learn_apply_to=msg_only`、E11 的 `edge_msg_gate_renorm=false`、E21 的 `label_budget`。

[evidence/README.md](breakthrough/evidence/README.md) 索引已纳入 Git 的紧凑证据表；完整日志、数据与检查点是本地资产，
见 [DATA_ASSETS.md](../DATA_ASSETS.md)。计划中出现任务不等于任务已完成，训练运行器也不会核对缓存与当前代码是否一致。

历史预设保存在 [archive/configs/](../archive/configs/README.md)，可通过显式路径加载；旧预算下的排名不能直接替代当前实验结果。
