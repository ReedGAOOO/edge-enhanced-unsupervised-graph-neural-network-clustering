# 实验详述、阅读笔记与历史代码包

更新：2026-10-17。当前入口是[项目首页](../README.md)、[实验计划索引](../configs/breakthrough/README.md)与[已跟踪证据表](../configs/breakthrough/evidence/README.md)。本目录同时保存当前实验详述和历史阅读资料，使用时注意各自语境。

| 资料 | 如何使用 |
|---|---|
| [BREAKTHROUGH_E1_E22.md](BREAKTHROUGH_E1_E22.md) | E1–E22 详细实验结果、对照、诊断和结论限制；具体数值结合 evidence CSV 核查 |
| [EXPLAIN_DSE_original_zh.md](EXPLAIN_DSE_original_zh.md) / [English](EXPLAIN_DSE_original_en.md) | 原始 DSE 参考代码的训练与解码流程说明，不是所有当前边增强分支的规格文档 |
| [EXPERT_BRIEF_model_status_zh.md](EXPERT_BRIEF_model_status_zh.md) | 早期专家简报；其中 B45 等分支排名早于 10 月的骨干训练审计，不代表当前推荐 |
| [历史版本说明](../archive/docs/exp/versions/README.md) | V5、V12、B14–B17 等早期分支的说明和演化背景 |

## 2026 年 4 月代码包

下列目录保留当时打包内容、说明、manifest 和校验信息，README 原文不随当前项目更新：

- [expert_code_bundle_20260414/](expert_code_bundle_20260414/README.md)：分目录核心代码和预设。
- [expert_code_bundle_full_flat_20260414/](expert_code_bundle_full_flat_20260414/README.md)：扁平核心代码包。
- [expert_code_bundle_mergedfull_20260414/](expert_code_bundle_mergedfull_20260414/README.md)：完整合并包及原包说明。
- [expert_code_bundle_ultralight_20260414/](expert_code_bundle_ultralight_20260414/README.md)：精简阅读包。

这些包用于追溯，不是当前可直接替换主干的实现；EP1/EP2 已不作为当前方向。完整复现 E2 还需要 `results/breakthrough_v1/_code/` 中经过 warm-start 修补的专家工作副本；该目录被 Git 忽略，与这里的原始包不同。资产范围见 [DATA_ASSETS.md](../DATA_ASSETS.md)。

## 旧链接

本目录的 `V5_core.md`、`V12_core.md`、`B14_PATHA_only_v1.md`、`B15_PATHB_v12_hier.md`、`B16_PATHA_PATHB_v12_hier.md`、`B17_PATHA_on_V5.md` 是早期指向 `../exp/versions/` 的符号链接；原路径已迁入 archive，当前链接失效。请直接使用[归档版本目录](../archive/docs/exp/versions/README.md)。这些兼容遗留文件不表示当前受支持的实现。
