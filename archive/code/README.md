# 历史代码入口

更新：2026-10-17。这里保存已退出顶层工具列表的辅助程序与实验驱动。当前可用入口见 [tools/README.md](../../tools/README.md)，研究结论见[项目首页](../../README.md)。

| 目录 | 内容 |
|---|---|
| [historical_tools/](historical_tools/) | 旧分支对比、B30 诊断、参数扫描和早期 Urban 实验驱动 |
| [historical_scripts/](historical_scripts/) | baseline、A2 及其对比实验的历史 shell 包装 |
| [historical_builders/](historical_builders/README.md) | 旧 Urban、Entities、Fraud、Bitcoin 与边属性对照构建器 |

这些文件按原始用途保留，迁移后没有统一适配当前目录布局。部分程序仍硬编码旧 `tools/` 路径、已归档的 `configs/presets/` 文件或旧工作站数据位置；执行前先检查源码、导入路径和输入输出。当前环境见 [UV_ENVIRONMENTS.txt](../../UV_ENVIRONMENTS.txt)。

历史运行和排名的适用边界见[归档总说明](../README.md)。不要直接把旧包装脚本的短训练预算当作当前可比较的实验配置。
