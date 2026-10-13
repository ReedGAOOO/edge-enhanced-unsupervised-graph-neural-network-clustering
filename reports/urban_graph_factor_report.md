# Urban 城市图差异的 3+6 因素诊断报告

**目的。** 本报告解释为什么同一套 Urban
原始数据在不同城市上会形成质量差异明显的图，并说明这些差异如何影响后续
edge-aware GNN 实验设计。

# 核心结论

Urban
图质量的跨城市差异不是单一原因造成的。当前证据不支持简单归因，例如"只是原始数据源不同"或"只是城市形态不同"。更稳妥的解释是分层诊断：

1.  **三个一级因素**是数据与标签口径、图投影构建规则、可观测城市形态代理变量。
2.  **六个二级因素**是任务标签错配、类别/度基线膨胀、label-pair
    异配结构、空间混杂、endpoint feature 冗余、relation 粒度问题。
3.  WashingtonDC 是当前最干净的 anchor city；Singapore 是 edge-rich but
    label-limited；Tokyo 和 Paris 是 label-rich but edge-weak；Shanghai
    和 Boston 更适合作为 projection-risk diagnostic。

本报告是诊断性分析，不是因果识别。关于河流、地形、规划制度、地方经济等宏观解释，目前只能作为合理假设；若要证明，需要外部协变量和匹配/反事实构图实验。

# 诊断框架

3+6
框架把"已经可以用现有数据检验的因素"和"仍需要外部数据支持的假设"分开。一级因素回答数据和构图是否可用；二级因素回答为什么一个图在
edge probe 上看似有效，但未必真的提供了独立的 edge semantics。

![Figure 1. 3+6 factor taxonomy for cross-city Urban graph
differences.](/home/aitx/workspace/projects/edge-enhanced-unsupervised-graph-neural-network-clustering/reports/urban_graph_factor_report_files/figure-markdown/factor-taxonomy-1.png)

# 三个一级因素的证据

第一轮审计支持混合解释。不同城市的 label coverage 和 class balance
差异很大；projection-tail risk 也明显不同；城市形态代理变量，尤其建筑
footprint 异质性，与 edge_attr permutation sensitivity 有关系。

![Figure 2. Evidence for primary factors: label quality, projection-tail
risk, and urban-form
proxies.](/home/aitx/workspace/projects/edge-enhanced-unsupervised-graph-neural-network-clustering/reports/urban_graph_factor_report_files/figure-markdown/primary-evidence-1.png)

解读：WashingtonDC 的 edge 证据强，同时 projection-tail risk
不极端，因此适合作为主线 anchor。Singapore 的 edge 证据也强，但 label
coverage 偏低。Shanghai 和 Boston 的价值在于暴露 projection risk。Paris
是重要 negative case：标签很多，但 edge 证据弱。

# 六个二级因素的证据

第二轮审计说明，单独看 homophily 不够。所有城市的 same-label lift
都高于随机 node-label baseline，但这个 lift 并不能排序 edge_attr
的真实有用性。因此，edge semantics 应同时评估 **homophily** 和
**structured heterophily**。

![Figure 3. Task mismatch and heterophily
diagnostics.](/home/aitx/workspace/projects/edge-enhanced-unsupervised-graph-neural-network-clustering/reports/urban_graph_factor_report_files/figure-markdown/additional-evidence-1-1.png)

这张图改变了对若干城市的解读。Beijing 和 Shanghai 的 label-pair
结构强，但 edge-probe 证据并不强；Tokyo 和 Paris 的标签较充分，但
same-label 学习不足以让 edge_attr 变得有用。因此下一步 probe 不能只预测
same-label，还应加入 label-pair 或 relation-pair 目标。

![Figure 4. Confounding and redundancy
diagnostics.](/home/aitx/workspace/projects/edge-enhanced-unsupervised-graph-neural-network-clustering/reports/urban_graph_factor_report_files/figure-markdown/additional-evidence-2-1.png)

Endpoint redundancy 是 WashingtonDC 和 Sydney 必须控制的问题。这并不否定
edge_attr，但要求更严格的 ablation：比较
endpoint-only、endpoint+distance、endpoint+degree、endpoint+relation
bits、endpoint+full
edge_attr。空间控制也必要，因为若干城市中仅用距离就能达到非零的
same-label AUC。

# 城市类型划分

当前城市类型划分应指导下一轮 V4
构图。更合理的做法不是把所有城市都当成同质
benchmark，而是让不同城市承担不同诊断角色。

  ----------------------------------------------------------------------------
  城市           诊断角色        下一步用途                  Edge     Perm
                                                             gain     drop
  -------------- --------------- --------------------------- -------- --------
  WashingtonDC   主线 anchor     主模型与 ablation anchor    0.008    0.011

  Singapore      edge            优先做                      0.007    0.012
                 强但标签受限    V4，但不能单独作主结果               

  Bangkok        次级诊断候选    进入 V4 smoke test 候选     0.003    0.003

  Melbourne      次级诊断候选    进入 V4 smoke test 候选     0.002    0.003

  Sydney         次级诊断候选    进入 V4 smoke test 候选     0.002    0.002

  Tokyo          标签充分但 edge 保留为 label-rich 对照      0.000    0.002
                 弱                                                   

  Paris          标签充分但 edge 保留为 label-rich 对照      -0.002   0.000
                 弱                                                   

  Madrid         次级诊断候选    进入 V4 smoke test 候选     0.001    0.002

  Johannesburg   低优先级 /      暂不优先投入                -0.007   0.004
                 负对照                                               

  Chicago        低优先级 /      暂不优先投入                -0.002   0.002
                 负对照                                               

  Shanghai       投影风险诊断    用于 projection-risk stress 0.002    0.004
                                 test                                 

  Beijing        低优先级 /      暂不优先投入                -0.001   -0.001
                 负对照                                               

  Boston         投影风险诊断    用于 projection-risk stress -0.003   -0.005
                                 test                                 
  ----------------------------------------------------------------------------

# 后续实验建议

下一步实验应把这些诊断因素转化为可证伪的 control。

  ---------------------------------------------------------------------------------
  因素                    立即可做的 control / test  用途
  ----------------------- -------------------------- ------------------------------
  数据 / 标签口径         每个城市报告 known-label   避免对弱标签城市过度解释
                          ratio、entropy、dominant   
                          class                      

  图投影构建              改变 entity cap、local     区分语义边和 projection
                          top-k、geometry fallback   artifact

  城市形态                按 building-area           检验形态是否调节 edge
                          tail、plot area、density   usefulness
                          proxy 分层                 

  任务标签错配            增加 label-pair 和         避免把功能性跨类边误判为失败
                          heterophily probe          

  类别 / 度基线           对比 node-label baseline   量化超越平凡类别结构的真实
                          和 edge-endpoint baseline  lift

  空间混杂                加 endpoint+distance 与    检验 edge_attr 是否超越
                          spatial block split        geography

  Endpoint 冗余           用 endpoint features 预测  检验独立 edge 信息
                          edge_attr；消融冗余特征    

  Relation 粒度           做 relation-specific probe 找出真正有用的 relation type
                          和 relation-drop ablation  

  Topology-degree 混杂    加 endpoint+degree control 检验 graph position
                          和 degree-preserving       是否解释增益
                          rewiring                   
  ---------------------------------------------------------------------------------

# 结论

当前最强的结论不是"所有 Urban
城市图都同样有意义"。更强、也更适合写成论文的结论是：Urban graph
validity 是**有条件的**。有些城市提供可用 edge semantics，有些城市暴露
label 或 projection 限制，有些城市则揭示了 same-label clustering probe
会低估的 heterophilous urban
relations。这个数据集是有价值的，但应被定位为跨城市诊断研究，而不是简单
benchmark。
