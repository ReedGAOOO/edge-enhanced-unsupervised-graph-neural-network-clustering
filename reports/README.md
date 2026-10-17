# Urban 数据研究报告

更新：2026-10-17。本目录保存城市图构建与有效性分析报告。当前聚类、分区选型及少标签实验请从[项目首页](../README.md)、[E1–E22 详细记录](../reading/BREAKTHROUGH_E1_E22.md)和[证据表](../configs/breakthrough/evidence/README.md)进入。

| 报告 | 可阅读版本 | 输入与范围 |
|---|---|---|
| [城市图差异 3+6 因素诊断](urban_graph_factor_report.Rmd) | [Markdown](urban_graph_factor_report.md) / [PDF](urban_graph_factor_report.pdf) | 读取 `results/data_construction_series/urban_all_city_exploration/` 下的城市诊断、优先级、差异驱动因素与二级因素 CSV |
| [Urban 因素实验包](urban_factor_experiment_packages.Rmd) | [Markdown](urban_factor_experiment_packages.md) / [PDF](urban_factor_experiment_packages.pdf) | 读取 `results/urban_factor_experiments/` 下的 label / projection / urban-form 汇总、决策矩阵、manifest 和图件 |

报告保留生成时的分析语境。早期模型对比存在训练轮数不足、部分运行把 `unknown` 算作类别等问题，其排名不能代替当前聚类结论。当前已知标签评价也不支持把 Urban 土地利用标签直接当作社区恢复真值；源数据质量、投影规则、空间混杂及城市差异仍是有效的研究方向。

## 重新生成

从仓库根目录运行，并先按[数据说明](../data/README.md)准备原始城市与预处理图。`results/` 被 Git 忽略，克隆仓库不会取得源实验输出；渲染脚本只读取已有结果，不会补跑缺失实验。

因素实验的历史编排入口是 [run_urban_factor_experiment_packages.py](../tools/run_urban_factor_experiment_packages.py)。先查看参数，并检查源码中的运行时配置：

```bash
.venv/bin/python tools/run_urban_factor_experiment_packages.py --help
```

该编排器仍硬编码已移除的 `gnn` Python 环境、`conda run -n r Rscript` 和旧工作站数据路径；其 projection 子程序也保留历史运行时配置。实际执行前须修正这些配置，或在当前环境中分别调用子任务，并显式设置数据根目录。`--dry_run true` 只向部分子任务传递预览参数，顶层仍会写 manifest/账本并执行其他子任务与 R 命令，不能作为全局只读预览。

两份报告所需结果完整后，可单独渲染：

```bash
Rscript tools/render_urban_factor_experiment_packages.R --results_root results/urban_factor_experiments
Rscript tools/render_urban_factor_report.R
```

第一份报告还依赖 all-city exploration 分析链的输出，相关入口是 [analyze_urban_raw_all_cities.py](../tools/analyze_urban_raw_all_cities.py)、[analyze_urban_city_difference_drivers.py](../tools/analyze_urban_city_difference_drivers.py)及 [analyze_urban_additional_difference_factors.py](../tools/analyze_urban_additional_difference_factors.py)。使用前检查各脚本的 `--help` 与上游诊断/探针输入。

[run_urban_data_paper_pipeline.py](../tools/run_urban_data_paper_pipeline.py) 是另一个 tile-level 数据机制分析入口，默认写到 `results/urban_data_paper/`，不能替代上述两份 Rmd 的全部输入。相关工具导航见 [tools/README.md](../tools/README.md)。

R 渲染需要 `rmarkdown`、`knitr`、`dplyr`、`readr`、`tidyr`，以及报告使用的 `ggplot2`、`ggrepel`、`patchwork`、`scales`、`jsonlite`；PDF 导出使用系统 `pandoc`、`typst` 和 Noto CJK 字体。根项目 Python 环境不会安装这些 R/系统依赖。

部分生成 Markdown 的图片链接仍是原工作站的绝对路径。已跟踪 PDF 提供可移植的阅读版本；需要更新报告时，从 Rmd 和本地实验结果重新渲染。本次导航更新保留报告正文、PDF 和原始图件的历史内容。
