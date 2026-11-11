# GT-SNT 对比算法（AAAI'26）

论文：Zhang, Li, Zhu, Zhong, Chen. *GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization.*
AAAI 2026（arXiv:2504.11840，v1 名为 GT-SVQ）。官方代码：github.com/Zhhuizhe/GT-SNT，提交号见 `UPSTREAM.txt`。
仓库没有许可证文件，代码仅用于学术对比。`official/` 中是未改动的原文件：模型（`nn/`）、训练脚本（`node_cls.py`、`node_cls_subgraph.py`）和各数据集配置（`config/*.yaml`）。

启动：`bash run_gtsnt.sh <dataset>`。依赖：

- `torch_geometric`；
- `spikingjelly==0.0.0.0.14`（官方代码用 `spikingjelly.clock_driven`，新版已改名；脚本启动时会检查）；
- `../common/protocol.py`（与 GCFormer / HubGT / SwapGT / Polynormer 相同）。

`torch_sparse` 可选：没有安装时，脚本用一个占位模块让 `official/utils.py` 能够导入。

## 1. 与其他对比算法一致的协议

| 项 | 取值 |
|---|---|
| 数据 | `${DATA_ROOT}/{adj.npz,feat.npy,label.npy,train.npy,val.npy,test.npy}`，读取代码与其他基线相同 |
| 运行次数 | `RUNS=5`，种子 42..46；`SPLIT_SUFFIXES` 可按划分循环 |
| 轮数上限 | `max(1000, 官方 epochs)`：其他数据集 1000，ogbn-arxiv / arxiv_year / ogbn-products / pokec 为 2000。全图流程 1 轮 = 1 次参数更新；子图流程 1 轮 = 遍历一遍训练节点（每块 512 个） |
| 早停 | 第 1 轮、每 `EVAL_EVERY=5` 轮及最后一轮验证；连续 `PATIENCE=30` 次检查不提升即停 |
| 模型选择 | 验证集最优的 checkpoint；测试集只评估一次 |
| 指标 | minesweeper / tolokers / questions 用 ROC-AUC（选模型、逐次输出、均值±标准差），其余用 Accuracy |
| 效率输出 | Params、Train time/epoch (s)、Time to best (s)、Peak GPU mem (MB)（另有 Tepoch、Mem.） |

官方协议对照：

| 项 | 官方做法 |
|---|---|
| 运行与种子 | 每个配置 1 次运行，种子 123 |
| 验证与早停 | 每轮都验证；连续 200 轮不提升即停 |
| 测试集 | 每轮都计算测试集，报告验证最优那一轮的测试值 |
| 划分（`dataset.py`） | Cora / Citeseer / Pubmed：每类 20 / 500 / 1000；CS / Physics：随机 10% / 10% / 80%；OGB：官方划分 |

本目录统一改为上表的项目协议和项目自己的划分文件。

## 2. 保留的官方做法

- **模型**：直接调用 `official/nn` 中的 `GTSNT`，参数名与 yaml 的 key 一致。它由三部分组成：
  - GCN 分支；
  - 脉冲节点分词器 SNT：每个节点有可学习的随机特征（N × cb_channels），在图上传播 T−1 步，经 PLIF 神经元得到脉冲序列，所有不同的脉冲模式构成码本；
  - 码本引导的注意力。
- **边**：`utils.precalculate_edge_attr(edge_index, N, "default")`，即每个节点加一个自环，边权全为 1（与 `dataset.py` 相同）。
- **特征**：Amazon（Computers、Photo）和 Coauthor 用 PyG `NormalizeFeatures`，其余用原始特征（`dataset.py`）。
- **训练**：Adam，交叉熵损失（`functional.compute_loss`）。
- **子图流程**（ogbn-products / pokec，对应 `node_cls_subgraph.py`）：
  - 每轮打乱训练节点，按 `batch_size` 分块；
  - GCN 分支使用每块目标节点在加自环图上的 k 跳入向子图（k = 层数）；
  - 分词器在原始全图（不加自环）上运行；
  - 验证 / 测试的块大小与训练相同。

## 3. 超参数来源与核对

**论文没有逐数据集的超参数表**，仓库的 `config/*.yaml`（cora、citeseer、pubmed、cs、physics、ogbn-arxiv、ogbn-products）是唯一的官方设置。

`run_gtsnt.sh` 中这 7 个配置的全部模型和优化参数都已与 yaml 自动逐项核对，结果一致：

- 核对的参数：hid、cb、层数、头数、T、neuron、v_threshold、dropout、attn_dropout、maximum_codes_num、normalize、residual、norm、lr、wd、epochs、batch_size。
- ogbn-arxiv.yaml 中的 `batch_size: 4096` 只被 `node_cls_subgraph.py` 读取；README 用 `node_cls.py`（全图）运行 arxiv，所以不使用它。

| 数据集 | 来源 | 说明 |
|---|---|---|
| cora, pubmed | official:config/cora.yaml / pubmed.yaml | |
| coauthor_cs | official:config/cs.yaml | NormalizeFeatures |
| ogbn_arxiv | official:config/ogbn-arxiv.yaml | 全图 |
| ogbn_products | official:config/ogbn-products.yaml | 子图，batch 512，maximum_codes_num 64 |
| amazon_photo, amazon_computers, corafull, facebook_large | fallback:cs | NormalizeFeatures（与官方 Amazon 加载方式相同） |
| wikics, roman_empire, amazon_ratings, minesweeper, tolokers, questions | fallback:cs | 原始特征 |
| arxiv_year | fallback:ogbn-arxiv | 同一张图、同一特征 |
| pokec | fallback:ogbn-products+batch2048 | 子图流程：163 万节点无法使用全图脚本；batch 改为 2048，理由见表后 |
| lastfm_asia | fallback:cs | 同 facebook_large（以人为节点、二值属性的社交网络，与 Coauthor 同类），NormalizeFeatures |
| github | fallback:config/physics.yaml | 社交网络，规模与平均度接近 Physics（37.7k 节点、平均度 15.3 vs 34.5k、14.4），NormalizeFeatures |
| dblp | fallback:config/pubmed.yaml | 同 pubmed（规模相近的引文网络），原始特征 |

选择 fallback 的理由：

- **cs 配置**是唯一同时开启残差和 BatchNorm 的中等规模配置。
- **异配数据集**：`dataset.py` 里有 Actor / Deezer / WebKB 的加载代码，但仓库没有给出这些数据集的配置。所以异配数据集也只能先用 cs 配置。
- **pokec 的 batch 2048**：其余参数沿用 products 配置，只把 batch 从 512 改为 2048，使每轮的优化步数与官方设置相当。
  - 官方 products：196,615 个训练节点 / 512 ≈ 384 步/轮；pokec：816,320 个训练节点 / 2048 ≈ 399 步/轮。若用 512，每轮要 1,594 步。
  - 子图流程的每一步都要在全图上运行分词器，GCN 用的 3 跳子图也覆盖了大部分边，所以单步开销接近整图，与 batch 大小关系不大。用 512 时，实测每轮训练约 45 分钟（共用的 A6000 上）。
  - 验证 / 测试的块大小仍与训练相同（官方做法），即 2048。
  - 如需改回，可用 `BATCH_SIZE=512 bash run_gtsnt.sh pokec`，来源标记会追加 `+BATCH_SIZE`。

fallback 只是起点，如需调参，只能依据验证集，例如 `LR=1e-3 T=3 bash run_gtsnt.sh roman_empire`。任何参数都可以用同名环境变量覆盖，覆盖项会追加在输出的 `hyperparameters:` 来源标记后面。

## 4. 参数量说明

Params 统计全部可训练参数，**包括 SNT 每层的逐节点随机特征（层数 × N × cb_channels）**。这部分随节点数线性增长：

| 数据集 | SNT 逐节点特征参数量 |
|---|---|
| ogbn-arxiv | 4 × 169,343 × 6 ≈ 4.1M |
| pokec | 3 × 1,632,803 × 8 ≈ 39.2M |
| ogbn-products | 3 × 2,449,029 × 8 ≈ 58.8M |

`summary.json` 每次运行的 `extra.snt_node_feature_weights` 记录这一部分，论文中可单独说明。

## 5. 工程改动（数值等价）

- 边以 edge-index 张量传入，GCNConv 和 SNT 的归一化与求和与官方 `SparseTensor` 输入相同，所以 torch_sparse 不是必需的。
- 例外是子图流程的 GCN 分支。按边传消息要生成 `[边数, hid]` 的张量：pokec 上一个 512 节点 batch 的 3 跳子图约 3,000 万条边，每个这样的张量就要 3.5 GiB。所以这里把子图作为 `torch.sparse` CSR 邻接矩阵 adj_t 传入，用 spmm 聚合（官方子图脚本同样传稀疏矩阵）。
  - 矩阵里已经为每个节点放好一个权重为 1 的自环：先去掉子图自带的自环，再给每个节点插入一个，结果与官方 `fill_diag`、edge-index 写法的 `add_remaining_self_loops` 相同。GCNConv 的 `add_self_loops` 因此关闭，`gcn_norm` 不必每层重新排序。
  - 加自环后的全图边按（目标，源）排序存放，k 跳子图的边保持这个顺序，所以每个 batch 构造 CSR 都不用排序。
  - SNT 仍用 edge-index：它的消息只有 cb_channels 维（pokec 约 1.4 GiB）。如果也用 CSR，CUDA 上 spmm 的反向要先把转置矩阵转回 CSR，相当于每次传播都对全图所有边排序一次（每个 batch 有 (T−1)×层数 = 6 次），训练会明显变慢。
  - 在服务器上用 `profile_subgraph.py` 可以逐 batch 对比各版本的训练、评估耗时和显存。
- 加速（数值与官方一致，`test_gtsnt.py` 逐项对比）：
  - SNT 每层要对 N 行脉冲做 `torch.unique(dim=0)`，这是逐列比较的排序。脉冲计数只取 0..T，把每行看作 T+1 进制数（第一列为最高位）后做一维 unique，得到的顺序、inverse 和 counts 都相同。官方文件不改，只在适配脚本里重新绑定 `SNT.forward`，除这一行外与官方代码相同。
  - 子图流程的 k 跳边改用逐跳扩展的节点掩码求出，不再调用 `k_hop_subgraph`。后者会对收集到的各跳列表（pokec 每个 batch 约 3,000 万个，含大量重复）做一次 unique，而结果并没有用到。
  - 评估时每层的分词器只算一次，所有 batch 复用。码本只取决于分词器参数和全图，与 batch 无关。
- 损失在训练设备上计算。官方 `node_cls.py` 先把 logits 移到 CPU。
- 每次前向后神经元膜电位都会重置，因此子图流程中节点分块的顺序不影响输出。`test_gtsnt.py` 对此有检查。

## 6. 自检

```bash
python test_gtsnt.py   # CPU 即可：边预处理与官方一致、重复前向无状态残留、子图稀疏输入与 edge-index 输入的 logits 和梯度一致（有 GPU 时也在 CUDA 上检查）、全图 / 子图两种流程端到端、ROC-AUC 数据集、效率输出、Params 含 SNT 特征
```
