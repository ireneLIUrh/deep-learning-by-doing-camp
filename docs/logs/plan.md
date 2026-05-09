# 学习计划

## 一、总体目标

目标是形成三种能力：

1. **理论能力**：能从线代、概率、统计、优化推导到机器学习和深度学习模型。
2. **代码能力**：能用 PyTorch / PyG 实现、训练、调试、评估模型。
3. **科研工程能力**：能把一个算法课题整理成 GitHub 项目，包括数据接口、模型模块、训练脚本、实验配置、结果复现、文档和论文复现记录。

推荐的学习顺序：

> **PyTorch 基础 → GNN 快速入门 → 动态/时空 GNN → 回补数学统计机器学习 → 深度学习系统化 → 大模型与 Agent 概念 → 课题模型原型工程化**

## 二、学习周期

分成 **10 个月主计划 + 长期迭代计划**。
强度按非工作时间设计：

| 时间       | 安排                                          |
| ---------- | --------------------------------------------- |
| 周一至周五 | 每晚 1–1.5 小时，主要用于读书、看课、做小练习 |
| 周六       | 3–4 小时，集中做代码任务和 GitHub 整理        |
| 周日       | 2 小时，复盘、写笔记、整理 README、规划下周   |
| 每周总量   | 8–12 小时                                     |

## 三、10个月 timetable

### Phase 0：环境与 GitHub 工程骨架搭建

**第 0–1 周**

目标：先把“学习—代码—笔记—实验”放进一个长期 GitHub 项目。

#### 需要建立的 GitHub 仓库

仓库名：

```text
deep-learning-by-doing-camp
```

初始结构：

```text
deep-learning-by-doing-camp/
├── README.md
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── .gitignore
├── configs/
│   ├── mlp_mnist.yaml
│   ├── gcn_cora.yaml
│   └── temporal_gnn_toy.yaml
├── notebooks/
│   ├── 00_linear_algebra_review.ipynb
│   ├── 01_probability_review.ipynb
│   └── 02_pytorch_basics.ipynb
├── src/
│   └── dlbydoing/
│       ├── __init__.py
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── utils/
│       └── visualization/
├── scripts/
│   ├── train_mlp.py
│   ├── train_gcn.py
│   └── train_temporal_gnn.py
├── tests/
│   ├── test_tensor_shapes.py
│   └── test_gcn_forward.py
├── reports/
│   ├── weekly_notes/
│   └── paper_reproduction/
└── results/
    └── .gitkeep
```

#### 本阶段 GitHub 任务

完成以下 5 个 commit：

```text
commit 1: initialize project structure
commit 2: add environment and dependency files
commit 3: add PyTorch tensor basics notebook
commit 4: add first train script for linear regression
commit 5: add README learning roadmap
```

#### 推荐资料

主线资料：

- PyTorch 官方 tutorials：用于 PyTorch 张量、Dataset、DataLoader、模型定义、训练循环入门。官方教程明确覆盖数据加载、构建神经网络、训练和保存模型等基础内容。([PyTorch Docs](https://docs.pytorch.org/tutorials/index.html?utm_source=chatgpt.com))
- PyTorch 官网生态页：PyTorch 当前生态中包含 PyTorch Geometric、Captum、skorch 等工具，其中 PyG 明确用于图、点云、流形等不规则数据。([PyTorch](https://pytorch.org/?utm_source=chatgpt.com))


### Phase 1：快速恢复数学与 PyTorch 基础

**第 1–4 周**

目标：恢复深度学习/GNN 所必需的数学和 PyTorch 能力。

#### Week 1：线代复健 + PyTorch Tensor

数学重点：

- 向量、矩阵、张量
- 矩阵乘法
- 线性变换
- 特征值/特征向量
- SVD 的直观含义
- 图邻接矩阵、度矩阵、拉普拉斯矩阵

代码任务：

1. 用 NumPy 实现矩阵乘法、标准化、PCA。
2. 用 PyTorch 重写。
3. 写一个 `notebooks/00_linear_algebra_review.ipynb`。
4. 写一个 `src/dlbydoing/utils/tensor_ops.py`。

课题连接：

```text
细胞表达矩阵 X: n_cells × n_genes
空间图邻接矩阵 A: n_cells × n_cells
GNN 输入: X + edge_index / adjacency
```

#### Week 2：概率复健 + 自动求导

数学重点：

- 随机变量
- 条件概率
- 期望、方差、协方差
- 常见分布：Bernoulli、Binomial、Gaussian、Poisson、Multinomial
- 最大似然估计
- 交叉熵与 KL divergence

代码任务：

1. 用 PyTorch 写 logistic regression。
2. 手写 binary cross entropy。
3. 比较手写梯度和 `torch.autograd`。
4. 写 `notebooks/01_probability_and_autograd.ipynb`。

课题连接：

细胞类型分类、通讯事件预测、边存在概率预测，本质上都离不开：

```text
P(y | x), P(edge | node_i, node_j), cross entropy, negative sampling
```

#### Week 3：优化基础 + MLP

数学重点：

- 梯度下降
- SGD / Momentum / Adam
- learning rate
- overfitting / underfitting
- L2 regularization
- train/validation/test split

代码任务：

1. 用 PyTorch 写 MLP 分类 MNIST 或 sklearn digits。
2. 完整实现：
   - Dataset
   - DataLoader
   - model
   - loss
   - optimizer
   - train loop
   - eval loop
   - checkpoint
3. 写 `scripts/train_mlp.py`。

#### Week 4：工程化基础

工程重点：

- `argparse` 或 `hydra/omegaconf`
- YAML config
- logging
- seed 固定
- checkpoint 保存
- TensorBoard / CSV log
- README 记录实验

代码任务：

1. 把 Week 3 的 MLP 改成命令行可运行：

```bash
python scripts/train_mlp.py --config configs/mlp_mnist.yaml
```

2. 加入：

```text
outputs/
├── checkpoints/
├── logs/
└── metrics.csv
```

本阶段结束产出：

```text
v0.1: PyTorch basic training pipeline
```

### Phase 2：迅速切入 GNN，服务课题

**第 5–10 周**

这是最关键阶段。先不追求所有深度学习分支都学完，而是快速具备 GNN 阅读论文和改代码能力。

#### 推荐主线资料

1. **Stanford CS224W: Machine Learning with Graphs**
   该课程专门讲图机器学习，公开课页面说明 lecture slides 和 assignments 会发布在线，适合作为 GNN 主线课程。([Stanford University](https://web.stanford.edu/class/cs224w/?utm_source=chatgpt.com))
2. **PyTorch Geometric 文档**
   PyG 是基于 PyTorch 的 GNN 库，官方文档说明它用于结构化数据上的 GNN 训练。([PyG Documentation](https://pytorch-geometric.readthedocs.io/?utm_source=chatgpt.com))
3. **PyG introduction by example**
   PyG 的 `Data(x, edge_index)` 是你之后处理细胞图最核心的数据结构，官方示例展示了如何用 `edge_index` 和节点特征构造一个图。([PyG Documentation](https://pytorch-geometric.readthedocs.io/en/2.6.1/get_started/introduction.html?utm_source=chatgpt.com))
4. **Open Graph Benchmark, OGB**
   OGB 是 Stanford SNAP 维护的大规模图学习 benchmark，提供真实、大规模、多任务数据集，并带统一 evaluator。([SNAP](https://snap-stanford.github.io/ogb-web/?utm_source=chatgpt.com))

#### Week 5：图数据结构与消息传递

数学重点：

- 图 $G=(V,E)$
- adjacency matrix
- degree matrix
- graph Laplacian
- message passing
- permutation invariance / equivariance
- neighborhood aggregation

代码任务：

1. 不用 PyG，手写一个最简单的 GCN layer：

```python
H = sigma(D^{-1/2} A D^{-1/2} X W)
```

1. 用 toy graph 测试 shape。
2. 写 `src/dlbydoing/models/gcn_from_scratch.py`。

课题连接：

把细胞看作 node：

```text
node = cell / spot
node feature = gene expression embedding, HVG expression, UCE embedding
edge = spatial adjacency / ligand-receptor edge / OT trajectory edge
```

#### Week 6：PyG 入门：GCN / GraphSAGE / GAT

模型重点：

- GCN
- GraphSAGE
- GAT

代码任务：

1. 用 PyG 跑 Cora / PubMed node classification。
2. 对比 GCN、GraphSAGE、GAT。
3. 保存实验表格：

```text
model | hidden_dim | layers | dropout | val_acc | test_acc
```

GitHub 任务：

```text
scripts/train_node_classifier.py
configs/gcn_cora.yaml
configs/gat_cora.yaml
reports/weekly_notes/week06_gnn_basics.md
```

#### Week 7：大图训练策略

你的数据有：

```text
node 数量 ~100k
node feature ~2000
edge 稀疏
time point ~10
```

所以必须学习：

- mini-batch graph training
- neighbor sampling
- subgraph sampling
- sparse tensor
- GPU memory profiling
- negative sampling

代码任务：

1. 用 PyG NeighborLoader 训练 GraphSAGE。
2. 记录显存/内存。
3. 写一个大图 toy benchmark。

课题连接：

你的真实数据不适合一上来 full-batch GAT。建议以后优先考虑：

```text
GraphSAGE / Cluster-GCN / GraphSAINT / neighbor sampling GAT
```

而不是直接把 10w 细胞的全图丢进 dense attention。

#### Week 8：链接预测与边分类

你的课题里“通讯分数”“细胞间相互作用”“OT 跨时间连接”天然适合设置为：

1. **edge prediction**
2. **edge classification**
3. **masked edge reconstruction**
4. **future edge prediction**

数学重点：

- link prediction
- negative sampling
- dot-product decoder
- bilinear decoder
- MLP edge decoder
- AUC / AP / F1

代码任务：

1. 在 citation graph 上做 link prediction。
2. 实现三种 decoder：

```text
dot product
bilinear
MLP([h_i, h_j, h_i*h_j])
```

课题连接：

可以对应你的数据：

```text
输入：time t 的空间图 + 表达特征
预测：time t+1 是否存在 OT 高概率连接 / 通讯增强事件
```

#### Week 9：动态图 / 时序图入门

重点阅读：

- EvolveGCN
- DySAT
- TGN
- Dynamic GNN survey

EvolveGCN 的核心思想是沿时间维度演化 GCN 参数，而不是依赖固定节点 embedding，适合动态图设置。([arXiv](https://arxiv.org/abs/1902.10191?utm_source=chatgpt.com))
DySAT 使用结构维度和时间维度的 self-attention 来学习动态图节点表示。([arXiv](https://arxiv.org/abs/1812.09430?utm_source=chatgpt.com))
TGN 面向连续时间动态图，把动态图表示为 timed events，并结合 memory module 和 graph operator 生成时间相关节点 embedding。([arXiv](https://arxiv.org/abs/2006.10637?utm_source=chatgpt.com))
2024–2025 的动态图 GNN survey 也指出，动态图 GNN 的核心是同时捕捉结构、时间和上下文关系，当前挑战包括可扩展性、异质信息和数据集不足。([arXiv](https://arxiv.org/abs/2405.00476?utm_source=chatgpt.com))

代码任务：

1. 构造 10 个时间点 toy graph。
2. 每个时间点训练一个 GCN。
3. 把每个时间点的 node embedding 输入 GRU / Transformer。
4. 预测下一个时间点的边或节点标签。

建议先实现这个 baseline：

```text
Static GNN encoder per time point + GRU temporal module + edge decoder
```

不要一开始就复现复杂 TGN。你的数据只有约 10 个时间点，更接近 discrete-time dynamic graph，而不是大规模 continuous event graph。

#### Week 10：课题原型 v0

建立一个与你课题强相关的最小原型：

```text
输入：
  X_t: 第 t 个时间点细胞表达特征
  A_spatial_t: 第 t 个时间点空间邻接图
  A_ot_t_to_t+1: 相邻时间点 OT 连接图

任务 A：
  masked gene expression prediction

任务 B：
  future edge prediction

任务 C：
  ligand-receptor communication score regression / classification
```

建议模型：

```text
Spatial GNN Encoder:
  X_t, A_spatial_t -> H_t

Temporal Transition Module:
  H_t + OT edges -> H_{t+1_pred}

Decoder:
  gene decoder: H_{t+1_pred} -> masked gene expression
  edge decoder: H_i, H_j -> edge probability
  communication decoder: H_i, H_j, LR features -> communication score
```

本阶段结束产出：

```text
v0.2: GNN baseline for dynamic cell graph
```


### Phase 3：经典机器学习与数理统计系统回补

**第 11–18 周**

这阶段的目标是把理论基础补扎实，尤其是为后面写论文、做实验设计、解释模型结果服务。

#### Week 11：经典数理统计 I

重点：

- 样本、总体、统计量
- 估计量
- bias / variance
- confidence interval
- hypothesis testing
- p-value
- multiple testing correction

课题连接：

通讯事件驱动基因、差异通讯边、显著性检验，都需要这部分。

代码任务：

1. 手写 t-test / permutation test。
2. 实现 Benjamini-Hochberg FDR。
3. 在模拟 ligand-receptor score 上做差异检验。

#### Week 12：经典数理统计 II

重点：

- linear regression
- logistic regression
- generalized linear model
- likelihood ratio test
- bootstrap
- permutation test

代码任务：

1. 用 NumPy 手写 linear regression。
2. 用 PyTorch 写 logistic regression。
3. 对比 sklearn 结果。

#### Week 13：机器学习 I

重点：

- kNN
- naive Bayes
- decision tree
- random forest
- SVM
- evaluation metrics

资料：

- 西瓜书：作为中文主线。
- sklearn 官方 examples：作为代码辅助。

代码任务：

1. sklearn 跑传统 ML pipeline。
2. 对单细胞元数据做 toy classification。
3. 输出 confusion matrix、ROC、PR curve。

#### Week 14：机器学习 II

重点：

- PCA
- t-SNE
- UMAP
- clustering
- GMM
- EM algorithm

课题连接：

单细胞分析里 PCA / UMAP / clustering 是基础工具，不能只会调用 Seurat/Scanpy，要理解背后的优化目标和距离结构。

#### Week 15–16：统计学习视角

重点：

- bias-variance tradeoff
- regularization
- cross-validation
- model selection
- feature selection
- calibration

代码任务：

1. 写一个统一的 `src/dlbydoing/evaluation/metrics.py`。
2. 写 `reports/weekly_notes/model_selection.md`。
3. 对 GNN 的 hidden_dim、layers、dropout 做小型 grid search。

#### Week 17–18：把统计检验接入 GNN 结果

课题导向任务：

```text
GNN 输出 edge score / communication score
↓
分组比较
↓
permutation test / bootstrap CI
↓
FDR correction
↓
筛选显著通讯边和驱动基因
```

产出：

```text
reports/prototype_statistical_testing_for_gnn_outputs.md
```

### Phase 4：深度学习系统化

**第 19–28 周**

这阶段开始系统读“花书”和现代教程。

#### 主线资料

- 《Deep Learning》花书：官方网页说明该书覆盖机器学习、深度学习、线性代数、概率、信息论、数值计算等背景内容。([深度学习书籍](https://www.deeplearningbook.org/?utm_source=chatgpt.com))
- MIT Press 页面也说明该书包含数学和概念背景，覆盖线性代数、概率与信息论、数值计算和机器学习。([MIT Press](https://mitpress.mit.edu/9780262035613/deep-learning/?utm_source=chatgpt.com))
- Dive into Deep Learning：建议作为代码化补充，尤其适合 PyTorch 实现。

#### Week 19：MLP 与反向传播

重点：

- computational graph
- backpropagation
- activation function
- initialization
- normalization

代码任务：

1. 手写两层 MLP 前向和反向。
2. 用 PyTorch 版本对照。

#### Week 20：CNN

重点：

- convolution
- pooling
- receptive field
- ResNet
- batch norm

代码任务：

1. CIFAR-10 ResNet mini version。
2. 工程化训练脚本。

与你课题关系：

CNN 不是主线，但空间转录组图像/空间模式建模可能会用到。

#### Week 21：RNN / GRU / LSTM

重点：

- sequence modeling
- hidden state
- vanishing gradient
- gated mechanism

课题连接：

你的时间点少，但 temporal module 仍可能用 GRU / LSTM 作为 baseline。

#### Week 22：Attention 与 Transformer

重点：

- query/key/value
- scaled dot-product attention
- multi-head attention
- positional encoding
- Transformer encoder

代码任务：

1. 手写 self-attention。
2. 用 Transformer 编码 toy time series。
3. 把 GNN embedding 序列输入 Transformer。

#### Week 23：Autoencoder / VAE

重点：

- representation learning
- encoder-decoder
- reconstruction loss
- variational inference
- ELBO

课题连接：

masked gene expression prediction 可以借鉴 autoencoder 思路。

#### Week 24：Contrastive Learning

重点：

- positive / negative pairs
- InfoNCE
- graph contrastive learning
- augmentation

课题连接：

可以为细胞 embedding 学习设计：

```text
same trajectory positive pair
near spatial neighbors positive pair
random far cells negative pair
```

#### Week 25–26：模型解释与可视化

重点：

- attention 不等于解释
- gradient-based attribution
- ablation
- perturbation
- embedding visualization

PyTorch 生态中的 Captum 是模型解释工具，PyTorch 官网生态页列出了 Captum 作为可解释性库。([PyTorch](https://pytorch.org/?utm_source=chatgpt.com))

课题连接：

你的最终论文需要回答：

```text
模型学到了什么？
哪些细胞通讯边重要？
哪些 ligand-receptor pair 重要？
哪些基因驱动了状态转变？
```

#### Week 27–28：复现一篇深度学习论文

建议不要选太复杂的论文。选以下之一：

```text
GCN
GraphSAGE
GAT
EvolveGCN
DySAT
TGN simplified version
```

产出：

```text
reports/paper_reproduction/gat_reproduction.md
src/dlbydoing/models/gat.py
scripts/train_gat.py
```


### Phase 5：动态图 GNN 与课题模型深化

**第 29–36 周**

这阶段回到你的课题主线。

#### Week 29：空间图构建策略

任务：

比较不同 edge 构建方式：

```text
kNN spatial graph
radius graph
Delaunay graph
ligand-receptor prior graph
OT trajectory graph
hybrid graph
```

代码产出：

```text
src/dlbydoing/data/graph_builder.py
```

#### Week 30：动态图表示方式

比较三种表示：

```text
Discrete snapshot graph:
  G_1, G_2, ..., G_T

Temporal interaction graph:
  events = (source, target, time, feature)

Multiplex heterogeneous graph:
  spatial edge + OT edge + LR edge
```

结合你的数据，第一版建议用：

```text
snapshot graph + OT inter-time edges
```

不要一开始就做连续时间 TGN。

#### Week 31：模型 baseline 设计

至少实现 4 个 baseline：

```text
Baseline 1: MLP only
Baseline 2: static GCN per time point
Baseline 3: GCN + GRU
Baseline 4: GAT + Transformer
```

#### Week 32：任务设计

建议设置三个任务，从简单到难：

| 任务                                   | 输入                     | 输出               | 意义           |
| -------------------------------------- | ------------------------ | ------------------ | -------------- |
| masked gene prediction                 | 部分基因 mask 后的细胞图 | 被 mask 的表达     | 学空间结构     |
| future embedding prediction            | time t 图                | time t+1 embedding | 学时间规律     |
| future edge / communication prediction | 细胞对 embedding         | OT/LR/通讯边概率   | 学通讯网络变化 |

#### Week 33：评价指标

分类任务：

```text
AUC, AP, F1, accuracy
```

回归任务：

```text
MSE, MAE, Pearson, Spearman
```

空间/生物学任务：

```text
cell type consistency
trajectory consistency
LR enrichment
known pathway recovery
```

#### Week 34：消融实验

至少做：

```text
without spatial edges
without OT edges
without attention
without temporal module
without LR prior
```

#### Week 35：可解释性分析

输出：

```text
important edges
important time windows
important LR pairs
important genes
cell-type-level communication heatmap
```

#### Week 36：整理成课题原型报告

产出：

```text
reports/project_prototype_v1.md
```

结构：

```text
1. Problem formulation
2. Data representation
3. Model architecture
4. Training objectives
5. Baselines
6. Evaluation metrics
7. Preliminary results
8. Next steps
```

### Phase 6：大模型、Agent、Skill 与科研自动化

**第 37–44 周**

这部分不作为前期主线，但后期必须学，因为它和求职、科研效率、工程能力都有关系。

#### Week 37：大模型基础

重点：

- language model
- tokenization
- pretraining
- instruction tuning
- RLHF / preference optimization
- embedding model

#### Week 38：Transformer 深化

重点：

- decoder-only Transformer
- causal mask
- KV cache
- scaling law 基本概念

#### Week 39：RAG

重点：

- embedding
- vector database
- chunking
- retrieval
- reranking
- citation grounding

科研应用：

建立一个：

```text
paper-rag-assistant
```

用于管理你课题相关论文。

#### Week 40：Agent

重点：

- tool use
- planning
- memory
- reflection
- multi-agent workflow
- evaluation

科研应用：

```text
agent 自动整理论文
agent 自动生成实验配置
agent 自动读取训练 log 并总结结果
```

#### Week 41：Skill / Workflow

重点：

- 把重复任务封装为技能
- 例如：
  - h5ad → graph data
  - Seurat metadata → PyG data
  - training log → plot
  - paper PDF → method summary

#### Week 42–44：做一个科研助手小项目

项目名：

```text
bio-gnn-research-assistant
```

功能：

```text
1. 输入论文 PDF 或摘要
2. 提取模型结构
3. 生成复现 checklist
4. 生成 PyTorch/PyG skeleton
5. 记录到 markdown
```

## 四、每周固定学习模板

每周不要“只看课”。建议固定成 5 个模块：

| 模块          | 时间     | 产出                 |
| ------------- | -------- | -------------------- |
| 数学/理论     | 2 小时   | 1 页公式笔记         |
| 纸质教材      | 1.5 小时 | 章节摘要             |
| 视频/网络课   | 1.5 小时 | 课程笔记             |
| PyTorch 代码  | 3–4 小时 | 可运行脚本           |
| GitHub 工程化 | 1–2 小时 | commit + README 更新 |

每周结束必须有：

```text
1. 一个 notebook
2. 一个 src/ 模块或 scripts/ 脚本
3. 一个 reports/weekly_notes/weekXX.md
4. 至少 3 个 Git commits
```

## 五、主线参考资料清单

### 数学基础

#### 主资料

1. Gilbert Strang 线性代数课程 / 教材
   用于恢复矩阵、特征值、SVD、最小二乘。
2. 《概率论与数理统计》本科教材
   用于恢复分布、估计、检验。
3. 花书前几章
   花书官方页面说明其覆盖线性代数、概率、信息论、数值计算和机器学习背景。([深度学习书籍](https://www.deeplearningbook.org/?utm_source=chatgpt.com))

#### 辅助资料

- 3Blue1Brown：线性代数直观理解。
- StatQuest：统计、机器学习、深度学习概念解释。
- Khan Academy：概率统计基础查漏补缺。

------

### 机器学习

#### 主资料

1. 周志华《机器学习》西瓜书
   适合中文系统回顾。
2. 李航《统计学习方法》
   适合补充数学推导。
3. sklearn 官方 examples
   用于传统机器学习代码练习。

#### 重点章节

```text
线性模型
决策树
支持向量机
贝叶斯分类器
集成学习
聚类
降维
特征选择
模型评估
```

------

### 深度学习

#### 主资料

1. PyTorch 官方教程
   用于掌握 PyTorch 基础训练流程。([PyTorch Docs](https://docs.pytorch.org/tutorials/index.html?utm_source=chatgpt.com))
2. 花书
   用于系统理论。
3. Dive into Deep Learning
   用于边学边写 PyTorch。

#### 重点章节

```text
MLP
Backpropagation
Optimization
CNN
RNN
Attention
Transformer
Autoencoder
Representation learning
Regularization
```

------

### 图神经网络

#### 主资料

1. Stanford CS224W
   课程公开资料包括 slides 和 assignments，适合作为 GNN 系统学习主线。([Stanford University](https://web.stanford.edu/class/cs224w/?utm_source=chatgpt.com))
2. PyTorch Geometric 文档
   PyG 是基于 PyTorch 的 GNN 库，用于结构化数据上的图神经网络。([PyG Documentation](https://pytorch-geometric.readthedocs.io/?utm_source=chatgpt.com))
3. OGB
   OGB 提供真实、大规模、多任务图学习 benchmark 和统一 evaluator。([SNAP](https://snap-stanford.github.io/ogb-web/?utm_source=chatgpt.com))

#### 辅助资料

1. DGL 文档
   DGL 支持 PyTorch，并强调可扩展到数亿节点和边的图训练。([dgl.ai](https://www.dgl.ai/dgl_docs/?utm_source=chatgpt.com))
2. GraphGym
   GraphGym 提供用于设计和评估 GNN 的平台，PyG 目前也支持 GraphGym。([PyG Documentation](https://pytorch-geometric.readthedocs.io/en/2.5.2/advanced/graphgym.html?utm_source=chatgpt.com))
3. PyTorch Lightning
   Lightning Trainer 可以抽象训练/验证/测试循环、callback、设备放置等工程细节。([Lightning AI](https://lightning.ai/docs/pytorch/stable//common/trainer.html?utm_source=chatgpt.com))

## 六、论文复现路线

按“由浅入深”复现，而不是一开始复现最新复杂模型。

### Level 1：基础 GNN

| 论文/模型 | 目标                                 |
| --------- | ------------------------------------ |
| GCN       | 理解邻接矩阵归一化与 message passing |
| GraphSAGE | 理解大图采样和 inductive learning    |
| GAT       | 理解 attention on graph              |

### Level 2：动态图 GNN

| 论文/模型 | 目标                                             |
| --------- | ------------------------------------------------ |
| EvolveGCN | 理解 GNN 参数随时间演化                          |
| DySAT     | 理解结构 attention + 时间 attention              |
| TGN       | 理解 event-based temporal graph 和 memory module |

### Level 3：与课题相关的自定义模型

最终复现/改造方向：

```text
Spatial GNN + Temporal module + OT inter-time edges + communication decoder
```

## 七、GitHub 边练边学任务设计

GitHub 项目分成 8 个 milestone。

### Milestone 1：PyTorch Basic

```text
目标：完成 PyTorch 基础训练框架
任务：
  - linear regression
  - logistic regression
  - MLP
  - checkpoint
  - config
  - README
```

### Milestone 2：Machine Learning from Scratch

```text
目标：恢复传统 ML 和统计基础
任务：
  - PCA from scratch
  - logistic regression from scratch
  - kmeans from scratch
  - bootstrap
  - permutation test
  - FDR correction
```

### Milestone 3：GNN from Scratch

```text
目标：不依赖 PyG 手写基础 GNN
任务：
  - graph data structure
  - GCN layer
  - GraphSAGE aggregation
  - GAT attention
  - toy graph classification
```

### Milestone 4：PyG GNN Pipeline

```text
目标：掌握 PyG 工程化训练
任务：
  - Cora node classification
  - PubMed node classification
  - link prediction
  - NeighborLoader large graph training
```

### Milestone 5：Dynamic GNN

```text
目标：动态图建模
任务：
  - snapshot graph dataset
  - GCN + GRU
  - GAT + Transformer
  - future edge prediction
```

### Milestone 6：Bio-GNN Prototype

```text
目标：接近课题真实需求
任务：
  - h5ad / Seurat metadata 转 graph
  - spatial edge builder
  - OT edge loader
  - masked gene prediction
  - communication edge prediction
```

### Milestone 7：Experiment Management

```text
目标：工程化科研实验
任务：
  - YAML config
  - random seed
  - logging
  - checkpoint
  - metrics.csv
  - plots
  - ablation table
```

### Milestone 8：Paper-style Report

```text
目标：把项目整理成可展示成果
任务：
  - method.md
  - experiments.md
  - reproduction.md
  - figures/
  - preliminary result summary
```

## 八、与课题最直接相关的 GNN 学习重点

数据结构是：

```text
每个时间点一张图
节点规模大：~100k
边稀疏
节点特征高维：~2000
时间点少：~10
相邻时间点有 OT 连边
```

因此不应该优先学“普通小图分类”，而应该重点学：

### 1. 大图训练

必须掌握：

```text
Neighbor sampling
mini-batch training
sparse adjacency
edge mini-batch
negative sampling
```

### 2. 动态图建模

你的数据更像：

```text
discrete-time dynamic graph
```

而不是高频 continuous-time event graph。
所以第一版模型建议：

```text
每个时间点用 GNN 编码空间图
时间维度用 GRU / Transformer 编码
OT 边用于跨时间信息传播或监督任务
```

### 3. 多关系图

你的边可能包括：

```text
spatial edge
OT trajectory edge
ligand-receptor prior edge
cell-type communication edge
```

所以后期要学：

```text
heterogeneous graph
relational GNN
edge type embedding
attention over edge types
```

### 4. 任务设计

最适合你课题的任务不是“普通节点分类”，而是：

```text
masked gene expression reconstruction
future cell-state prediction
future edge prediction
communication score prediction
driver gene identification
```

## 九、每月实际执行表

### 第 1 月：PyTorch + 数学复健

| 周     | 理论       | 代码                | GitHub 产出                      |
| ------ | ---------- | ------------------- | -------------------------------- |
| Week 1 | 线代复习   | tensor / PCA        | `00_linear_algebra_review.ipynb` |
| Week 2 | 概率 + MLE | logistic regression | `01_probability_autograd.ipynb`  |
| Week 3 | 优化       | MLP                 | `train_mlp.py`                   |
| Week 4 | 工程化     | config + logging    | `v0.1`                           |

### 第 2 月：GNN 快速入门

| 周     | 理论            | 代码                    | GitHub 产出                |
| ------ | --------------- | ----------------------- | -------------------------- |
| Week 5 | message passing | GCN from scratch        | `gcn_from_scratch.py`      |
| Week 6 | GCN/GAT/SAGE    | PyG node classification | `train_node_classifier.py` |
| Week 7 | 大图采样        | NeighborLoader          | `large_graph_training.md`  |
| Week 8 | link prediction | edge decoder            | `train_link_prediction.py` |

### 第 3 月：动态图 GNN + 课题原型

| 周      | 理论                | 代码                   | GitHub 产出               |
| ------- | ------------------- | ---------------------- | ------------------------- |
| Week 9  | EvolveGCN/DySAT/TGN | toy temporal graph     | `temporal_dataset.py`     |
| Week 10 | GCN+GRU             | future edge prediction | `train_temporal_gnn.py`   |
| Week 11 | 统计检验            | permutation/FDR        | `stats_tests.py`          |
| Week 12 | 课题抽象            | Bio-GNN prototype      | `project_prototype_v0.md` |

### 第 4–5 月：统计与传统 ML 回补

重点：

```text
统计检验
线性模型
SVM
随机森林
PCA/UMAP
聚类
模型评估
```

产出：

```text
ml_from_scratch/
evaluation/
reports/classical_ml_review.md
```

### 第 6–7 月：深度学习系统化

重点：

```text
CNN
RNN
Attention
Transformer
VAE
Contrastive learning
Interpretability
```

产出：

```text
attention_from_scratch.py
transformer_encoder.py
graph_contrastive_learning.md
```

### 第 8–9 月：动态图 GNN 深化

重点：

```text
heterogeneous graph
multi-edge-type graph
dynamic graph
ablation
large graph training
```

产出：

```text
bio_gnn_prototype_v1
ablation_results.csv
```

### 第 10 月：大模型 / Agent / Skill

重点：

```text
LLM
RAG
Agent
Tool use
科研自动化 workflow
```

产出：

```text
paper_rag_assistant
bio_gnn_research_assistant
```

## 十、最先应该做的 14 天任务

为了马上启动，建议你先按这个做。

### Day 1

创建仓库：

```bash
mkdir deep-learning-by-doing-pytorch
cd deep-learning-by-doing-pytorch
git init
```

建立目录结构，写 README。

### Day 2

创建 conda 环境：

```bash
conda create -n dl-gpu python=3.10 -y
conda activate dl-gpu
pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy pandas scikit-learn matplotlib seaborn jupyter pyyaml tqdm
```

具体 PyTorch / PyG 安装命令以后要根据你的集群 CUDA 版本调整。

### Day 3–4

完成 PyTorch tensor notebook：

```text
tensor 创建
reshape
matmul
broadcast
autograd
GPU tensor
```

### Day 5–6

写 linear regression from scratch：

```text
NumPy version
PyTorch manual version
PyTorch nn.Module version
```

### Day 7

写第一篇周报：

```text
reports/weekly_notes/week01.md
```

模板：

```markdown
# Week 01 Notes

## 本周目标
## 数学复习
## 代码实现
## 遇到的问题
## 和课题的连接
## 下周计划
```

### Day 8–9

复习概率、MLE、cross entropy。

### Day 10–11

写 logistic regression。

### Day 12–13

写 MLP MNIST / digits 分类。

### Day 14

整理 README，打 tag：

```bash
git tag v0.1-pytorch-basics
```

## 十一、学习方式建议

你最适合的学习方式是：

```text
先有问题 → 找理论 → 写代码 → 跑实验 → 写报告 → 回到课题
```

不要用：

```text
先完整看完一本书 → 再开始写代码
```

因为这会拖太久，也不能及时辅助课题。

每个知识点都按四步走：

```text
1. 这个算法解决什么问题？
2. 数学公式是什么？
3. PyTorch 怎么实现？
4. 能不能迁移到我的细胞图数据？
```

例如 GAT：

```text
问题：邻居重要性不同
数学：attention coefficient α_ij
代码：GATConv / self-attention on edges
课题：不同空间邻居、不同 LR pair 对细胞状态影响不同
```

## 十二、后续如何监督

后续我们可以按“每周一次学习组会”的方式推进。你每周发给我：

```text
1. 本周学了哪些章节/视频
2. GitHub commit 截图或 commit log
3. 运行结果或报错
4. 本周最不懂的 3 个问题
5. 下周想完成什么
```

我可以帮你做：

```text
代码 review
数学推导讲解
论文拆解
实验设计
GitHub 项目结构调整
README/报告润色
课题模型方案迭代
求职项目包装
```

你第一阶段的检查标准很简单：

> **第 1 个月结束时，你应该能独立写出一个 PyTorch 训练脚本。
> 第 2 个月结束时，你应该能用 PyG 跑 GCN/GAT/GraphSAGE。
> 第 3 个月结束时，你应该有一个与你课题相关的动态图 GNN prototype。**

这就是最短路径。先让 GNN 能服务课题，再系统补齐深度学习全图。