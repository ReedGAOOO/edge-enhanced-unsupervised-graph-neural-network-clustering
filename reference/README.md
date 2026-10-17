# 参考代码与论文

更新：2026-10-17。本目录保存上游代码快照、论文和独立参考项目。当前实现位于仓库根目录；使用方式和研究结论以[项目首页](../README.md)及[实验记录](../reading/BREAKTHROUGH_E1_E22.md)为准。

| 入口 | 用途 |
|---|---|
| [DSE_clustering-main/](DSE_clustering-main/README.md) | 原始 DSE 发布代码快照；其 README、图件及 [LICENSE](DSE_clustering-main/LICENSE)保留原文 |
| [REFERENCE_DSE_original_merged.py](REFERENCE_DSE_original_merged.py) | 便于阅读的原始实现合并文件 |
| [2504.09970v2.pdf](2504.09970v2.pdf)、[sun24g.pdf](sun24g.pdf) | DSE / LSEnet 相关论文的本地副本 |
| `urbanity/` | 独立上游工作副本，在本仓库中记录为 Git link，不是根项目 Python 包 |

[DSE 原实现阅读说明](../reading/README.md)解释训练与解码数据流。参考快照用于追溯代码来源；它与当前边增强实现、论文严格复现之间的差别需要单独核对，不能直接用上游 README 的结果图替代本项目实验。

## Urbanity 的获取与环境

根项目环境不会安装或同步 `urbanity/`。它使用独立环境，见 [UV_ENVIRONMENTS.txt](../UV_ENVIRONMENTS.txt)。本仓库目前没有该 Git link 的 `.gitmodules` 映射，因此普通克隆后不能依靠 `git submodule update` 自动取得完整工作副本；需要使用 Urbanity 时，应从其上游另行获取，并核对记录的版本。

本机 Urbanity 内可能有独立的未提交修改，额外收集的参考目录也可能未被 Git 跟踪。它们不属于本次 README 提交所发布的资产，也不是根项目训练的必需输入。上游 README 与许可证按来源保留，不随项目导航一起改写。
