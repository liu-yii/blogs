---
date: '2026-05-19T10:00:00+08:00'
draft: false
title: 'MLLM'
---

*随着 LLM 能力的持续涌现，一个自然的问题浮现：语言模型能否真正"看懂"这个世界？多模态大模型（MLLM）正是这一追问的工程化答案。从 CLIP 的对比学习到 LLaVA 的简洁投影，从 Qwen2-VL 的原生分辨率到 GPT-4V 的惊艳表现，视觉与语言的融合正在重塑 AI 系统的能力边界。本文试图从数学原理出发，深度剖析 MLLM 的架构演化与核心技术。*

## 1. 引言：多模态大模型的范式转换

### 1.1 从纯文本 LLM 到多模态理解的演进

大语言模型（LLM）的成功建立在一个核心假设上：语言是知识表示的通用介质。通过在海量文本上进行自回归预训练，模型学会了词语之间的语义关联、因果推理模式以及世界知识的隐式编码。然而，这一范式存在根本性局限——**语言描述的是对现实世界的符号抽象，而非现实世界本身**。

人类认知并非单一模态的。视觉信息占人脑信息处理量的约 70%，语言理解本质上是多模态的——"苹果是红色的"这句话的语义，需要与视觉经验绑定才能真正"理解"，而非仅在符号层面完成映射。

多模态大模型（Multimodal Large Language Model, MLLM）的核心命题是：**将视觉感知能力注入 LLM 的语言推理框架**。这不是简单地将两个独立系统拼接，而是要建立视觉与语言之间深层的语义对齐，使模型能够在统一的表示空间中推理跨模态信息。

从历史演进来看，视觉-语言模型（VLM）经历了三代范式转变：

| 时代 | 代表工作 | 核心方法 | 局限 |
|------|---------|---------|------|
| **第一代**（2019-2020） | VisualBERT, ViLBERT, UNITER | BERT 双流融合，区域特征拼接 | 依赖 Faster R-CNN 区域提取，速度慢、端到端困难 |
| **第二代**（2021-2022） | CLIP, ALIGN, Florence | 大规模对比学习，双塔对齐 | 仅做检索/分类，不具备生成能力 |
| **第三代**（2023-至今） | LLaVA, InstructBLIP, Qwen-VL, GPT-4V | 视觉编码器 + LLM，端到端指令微调 | 高分辨率效率、视频时序建模等挑战 |

### 1.2 核心问题：如何让 LLM "看到"图像？

将视觉信息融入 LLM 的根本挑战在于**表示空间的异质性（heterogeneity）**：

- **语言空间**：离散的 token 序列，每个 token 在 $d_{\text{model}}$ 维嵌入空间中有精确的语义位置，通过 causal attention 建模序列依赖
- **视觉空间**：连续的像素矩阵，语义信息分散在空间维度上，具有平移不变性、尺度不变性等特有结构

将图像"翻译"为 LLM 可理解的形式，需要解决三个子问题：

1. **视觉特征提取**：如何从原始像素中提取语义丰富的特征表示？（→ Vision Encoder）
2. **跨模态对齐**：如何将视觉特征映射到与语言特征兼容的语义空间？（→ Alignment Module）
3. **多模态理解与生成**：LLM 如何在接收视觉信息后进行推理和文本生成？（→ 自回归 LLM）

### 1.3 两大范式：对齐融合 vs 原生多模态

当前 MLLM 存在两种主流架构范式：

**范式一：对齐融合（Modality Alignment）**

将预训练的视觉编码器（如 ViT-L/14）与预训练的 LLM（如 LLaMA-3）通过一个轻量化的"连接器"（Connector/Projector）桥接。视觉和语言各自在独立的表示空间内预训练，连接器负责跨模态对齐。

$$\underbrace{f_{\text{visual}}(I)}_{\text{Vision Encoder}} \xrightarrow{\text{Projector}} \underbrace{z_{\text{vis}} \oplus z_{\text{text}}}_{\text{统一 token 序列}} \xrightarrow{\text{LLM}} \text{Response}$$

代表：LLaVA, InstructBLIP, Qwen-VL, InternVL

**范式二：原生多模态（Early Fusion / Native Multimodal）**

从头开始训练统一的多模态 Transformer，不区分"视觉编码器"和"语言模型"，所有模态在同一个统一架构中联合学习表示。

代表：Gemini（部分）、CM3Leon、Chameleon、JetMoE-Instruct

两种范式的核心权衡：

| 维度 | 对齐融合 | 原生多模态 |
|------|---------|-----------|
| **训练成本** | 低（复用预训练组件） | 极高（从头训练） |
| **模态交互深度** | 浅（仅通过投影层） | 深（每一层都有跨模态交互） |
| **迁移能力** | 强（继承预训练知识） | 需重新学习所有知识 |
| **扩展灵活性** | 高（可替换编码器/LLM） | 低（架构改动代价大） |
| **视觉推理上界** | 受投影层表达能力限制 | 理论上限更高 |
| **当前主流** | ✓ | 探索阶段 |

目前工业界和学术界的主流仍是对齐融合范式，因其训练效率高、可复用强大的预训练 LLM，且通过精心设计的对齐策略已经取得了令人惊艳的效果。本文将以这一范式为主线展开深度分析。

---

## 2. 视觉编码器（Vision Encoder）

视觉编码器是 MLLM 的"眼睛"，负责将原始图像转化为语义密集的特征表示。现代 MLLM 普遍采用 Vision Transformer（ViT）架构作为视觉编码器，并通过大规模对比预训练（CLIP、SigLIP 等）获得强大的跨模态语义表示能力。

### 2.1 ViT：视觉 Transformer 原理推导

Vision Transformer（ViT, Dosovitskiy et al., 2021）的核心思想是将图像处理问题转化为序列建模问题——通过将图像分割为固定大小的 Patch，将每个 Patch 视为一个"视觉词元"，然后用标准 Transformer 处理这个序列。

**Patch Embedding 的数学形式**

给定输入图像 $I \in \mathbb{R}^{H \times W \times C}$（$H, W$ 为高宽，$C$ 为通道数），ViT 将其划分为 $N$ 个不重叠的 Patch，每个 Patch 大小为 $P \times P$：

$$N = \frac{H \times W}{P^2}$$

对每个 Patch $p_i \in \mathbb{R}^{P \times P \times C}$，通过线性投影映射到 $d$ 维嵌入空间：

$$z_i^{(0)} = p_i \cdot W_E + b_E, \quad W_E \in \mathbb{R}^{(P^2 C) \times d}$$

加入可学习的位置编码 $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times d}$ 和分类 token $[\text{CLS}] \in \mathbb{R}^d$，得到初始序列：

$$\mathbf{Z}^{(0)} = [z_{\text{cls}}; z_1^{(0)}; z_2^{(0)}; \ldots; z_N^{(0)}] + E_{\text{pos}}$$

**Transformer Block 的递推关系**

第 $l$ 层 Transformer Block 的计算如下：

$$\mathbf{Z}'^{(l)} = \text{MSA}(\text{LN}(\mathbf{Z}^{(l-1)})) + \mathbf{Z}^{(l-1)}$$

$$\mathbf{Z}^{(l)} = \text{MLP}(\text{LN}(\mathbf{Z}'^{(l)})) + \mathbf{Z}'^{(l)}$$

其中 MSA 为多头自注意力（Multi-Head Self-Attention），LN 为 Layer Normalization。

**自注意力的计算**：对于 $h$ 个头，每个头的计算为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

$$Q = \mathbf{Z} W^Q,\quad K = \mathbf{Z} W^K,\quad V = \mathbf{Z} W^V$$

其中 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$，$d_k = d / h$。

**视觉 token 数量与分辨率的关系**

这是 MLLM 设计中的一个关键约束。对于分辨率 $(H, W)$ 和 Patch 大小 $P$，视觉 token 数量为：

$$N_{\text{vis}} = \frac{H \times W}{P^2}$$

| 分辨率 | Patch=14 | Patch=16 | Patch=32 |
|--------|---------|---------|---------|
| 224×224 | 256 | 196 | 49 |
| 336×336 | 576 | 441 | 110 |
| 448×448 | 1024 | 784 | 196 |
| 672×672 | 2304 | 1764 | 441 |
| 1024×1024 | 5376 | 4096 | 1024 |

视觉 token 数量的二次增长特性是高分辨率 MLLM 面临的核心挑战——将分辨率翻倍，视觉 token 数量增加 4 倍，LLM 的注意力计算复杂度以 $O(N^2)$ 增长，显存和延迟开销快速膨胀。

### 2.2 CLIP 预训练：对比学习的图像-文本对齐

CLIP（Contrastive Language-Image Pre-Training, Radford et al., 2021）是现代 MLLM 视觉编码器的标志性工作。其核心思想是通过在大规模图像-文本对上进行对比学习，迫使视觉编码器学习与自然语言语义对齐的视觉表示。

**对比学习目标函数**

给定一个 batch 内的 $N$ 个图像-文本对 $\{(I_i, T_i)\}_{i=1}^N$，CLIP 分别用视觉编码器 $f_I$ 和文本编码器 $f_T$ 提取特征，然后计算归一化的特征向量：

$$v_i = \frac{f_I(I_i)}{\|f_I(I_i)\|_2}, \quad t_i = \frac{f_T(T_i)}{\|f_T(T_i)\|_2}$$

构造 $N \times N$ 的相似度矩阵 $\mathbf{S}$，其中 $S_{ij} = v_i \cdot t_j / \tau$，$\tau$ 为可学习的温度参数。

CLIP 的训练目标是**同时最大化图到文和文到图两个方向的对角线元素概率**：

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left[\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I}\right]$$

$$\mathcal{L}_{I \to T} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ij})}$$

$$\mathcal{L}_{T \to I} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ji})}$$

这本质上是一个 $N$-way 分类的 InfoNCE Loss（见 [第 3.1 节](#31-对比学习对齐contrastive-alignment)），正样本对 $(I_i, T_i)$ 的相似度需要在所有 $N$ 个样本中排名最高。

**双塔架构的数学分析**

CLIP 的双塔设计（Vision Encoder ⊥ Text Encoder，仅在最终表示层交互）有其深刻的数学动机：

- **独立性假设**：如果将两个塔的特征视为随机变量 $V$ 和 $T$，对比学习的目标等价于最大化 $V$ 和 $T$ 之间的互信息（Mutual Information）的下界：
  $$\mathcal{L}_{\text{InfoNCE}} \leq I(V; T)$$
  
- **负样本作为噪声分布**：每个 batch 内其他 $N-1$ 个文本描述构成了当前图像的"噪声分布"，InfoNCE 优化等价于在信道噪声环境中最大化信号传输效率

**Batch Size 与负样本质量的关系**

CLIP 对 batch size 的敏感性来自于 InfoNCE 的统计性质。设 batch size 为 $N$，则损失函数中的分母包含 $N-1$ 个负样本。当 $N$ 较小时：

1. **负样本估计偏差大**：$N-1$ 个负样本无法充分代表整个"噪声分布"，导致梯度估计不稳定
2. **假负样本（False Negative）问题**：batch 内可能出现语义相似但标注为负样本的对，污染梯度信号

理论上，InfoNCE 的方差与 $1/N$ 成正比——batch size 越大，梯度估计越稳定。CLIP 使用 32768 的超大 batch size，正是为了确保每个样本都有足够多的高质量负样本。这也解释了为什么 CLIP 需要数百个 GPU 才能训练——分布式训练中，每个 GPU 局部 batch 内的负样本不够多，必须通过 all-gather 操作收集所有设备上的特征，构造全局 batch 的相似度矩阵。

```python
# CLIP 对比损失的简化实现（分布式感知版本）
import torch
import torch.nn.functional as F
import torch.distributed as dist

def clip_loss(image_features, text_features, temperature, world_size=1):
    # image_features, text_features: [N, d] 已 L2 归一化
    
    if world_size > 1:
        # 跨 GPU 收集所有特征
        all_image = torch.cat(dist.all_gather_as_list(image_features), dim=0)
        all_text  = torch.cat(dist.all_gather_as_list(text_features),  dim=0)
    else:
        all_image, all_text = image_features, text_features
    
    N = all_image.shape[0]
    logits = (image_features @ all_text.T) / temperature  # [local_N, global_N]
    
    # 对角线为正样本（offset 是当前 rank 的起始 index）
    rank_offset = dist.get_rank() * image_features.shape[0] if world_size > 1 else 0
    labels = torch.arange(rank_offset, rank_offset + image_features.shape[0],
                          device=image_features.device)
    
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T[:N//world_size], labels - rank_offset)  # 简化
    return (loss_i2t + loss_t2i) / 2
```

### 2.3 SigLIP：Sigmoid Loss 的改进

SigLIP（Sigmoid Loss for Language-Image Pre-Training, Zhai et al., 2023）针对 CLIP softmax 归一化的局限性提出了改进。

**Softmax 的核心问题**：CLIP 的 softmax 归一化需要在整个 batch 维度上计算分母，这意味着：
1. 分母的计算需要 all-reduce 操作，在分布式训练中引入通信瓶颈
2. batch 内所有样本相互影响——一个"困难负样本"会拉高分母，影响所有正样本的梯度

**SigLIP 的 Sigmoid Loss**：将多类别 softmax 替换为每对独立的二分类 sigmoid：

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N \begin{cases} \log \sigma(\beta \cdot S_{ij} + b) & \text{if } i = j \\ \log \sigma(-\beta \cdot S_{ij} - b) & \text{if } i \neq j \end{cases}$$

其中 $\sigma$ 为 sigmoid 函数，$\beta > 0$ 为可学习的缩放参数，$b$ 为偏置（初始化为 $-10$，偏向负类）。

**数学上的关键区别**：

| 属性 | CLIP (Softmax) | SigLIP (Sigmoid) |
|------|---------------|-----------------|
| 分母归一化 | 全局 batch 归一化 | 无（每对独立） |
| 负样本交互 | 通过 softmax 耦合 | 独立优化 |
| 分布式通信 | all-reduce 必需 | 仅需 all-gather 特征 |
| 正负样本比例 | 隐式（由 batch 决定） | 显式（$N^2 - N$ 个负对） |
| batch size 敏感性 | 强（负样本质量随 batch 增大） | 弱（每对独立判断） |

SigLIP 通过将大 batch 的对比学习分解为独立的二元分类，避免了 softmax 的全局归一化依赖，在分布式训练中具有更好的可扩展性。实验显示，SigLIP 在相同 batch size 下通常优于 CLIP，且在 batch size 较小时优势更明显。

**SigLIP 中偏置 $b$ 的作用**：初始化 $b = -10$ 相当于告诉模型"默认情况下所有配对都是负样本"——这符合真实数据分布（一个 batch 中正样本只占 $1/N$）。随着训练进行，$b$ 会逐渐上升以适应实际的正样本概率。

### 2.4 EVA-CLIP 及其改进

EVA-CLIP（Wang et al., 2023）是目前 MLLM 领域最常用的视觉编码器之一，其核心创新是将 CLIP 的训练规模推向新高度，并通过以下改进提升训练稳定性：

**EVA 的掩码图像建模预训练**：在 CLIP 对比学习之前，先用掩码自编码器（MAE 风格）预训练 ViT，学习局部视觉特征。然后在 CLIP 阶段，模型同时优化对比目标和视觉特征重建目标：

$$\mathcal{L}_{\text{EVA}} = \mathcal{L}_{\text{CLIP}} + \lambda \cdot \mathcal{L}_{\text{MIM}}$$

其中 $\mathcal{L}_{\text{MIM}}$ 为 Masked Image Modeling 损失（预测被遮挡 patch 的特征）。这种混合训练让编码器既学习全局语义对齐，又保留细粒度的局部特征。

**主流视觉编码器对比**：

| 模型 | 架构 | 参数量 | 训练数据 | 预训练方式 | MLLM 中的典型使用 |
|------|------|--------|---------|-----------|-----------------|
| CLIP ViT-L/14 | ViT-L | 307M | LAION-400M | 对比学习 | LLaVA-1.0/1.5 |
| CLIP ViT-L/14@336px | ViT-L | 307M | LAION-400M | 对比学习 | LLaVA-1.5 |
| SigLIP-SO400M | ViT-400M | 400M | WebLI | SigLIP | LLaVA-1.6, Qwen2-VL |
| EVA-CLIP-18B | ViT-18B | 18B | LAION 等 | EVA + CLIP | InternVL2 |
| InternViT-6B | ViT-6B | 6B | 多源数据 | 对比 + 生成 | InternVL2 |
| DINOv2-ViT-L | ViT-L | 307M | LVD-142M | 自监督（DINO） | 部分模型 |

---

## 3. 图文对齐策略（Image-Text Alignment）

图文对齐是 MLLM 架构设计的核心——如何让视觉特征与语言模型的语义空间"说同一种语言"。这一问题有三种本质不同的解法。

### 3.1 对比学习对齐（Contrastive Alignment）

**InfoNCE Loss 的完整推导**

InfoNCE（Noise Contrastive Estimation，van den Oord et al., 2018）是理解 CLIP 等对比学习方法的数学基础。

设图像特征 $v$ 和文本特征 $t$ 通过联合分布 $p(v, t)$ 采样，而负样本文本特征来自噪声分布 $q(t)$（通常取边缘分布）。InfoNCE 的目标是训练一个评分函数 $f(v, t)$，使正样本对的得分高于噪声样本。

对于一个正样本 $t_{\text{pos}}$ 和 $K$ 个噪声样本 $\{t_k\}_{k=1}^K$，InfoNCE 损失为：

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(f(v, t_{\text{pos}}))}{\exp(f(v, t_{\text{pos}})) + \sum_{k=1}^K \exp(f(v, t_k))}\right]$$

**定理（InfoNCE 是互信息下界）**：设 $f^*(v, t) = \log \frac{p(t|v)}{q(t)} + C$，则：

$$I(V; T) \geq \log(K+1) - \mathcal{L}_{\text{InfoNCE}}$$

**证明**：

$$\mathcal{L}_{\text{InfoNCE}} \geq -\log(K+1) + \mathbb{E}\left[\log \frac{K \cdot p(t|v)}{q(t)}\right] \cdot (-1) + \ldots$$

（详细证明利用 Jensen 不等式和 KL 散度非负性）

这一理论结果表明：**当 batch size $K$ 增大时，互信息的估计下界更紧**，即 batch size 越大，对比学习越接近真正最大化模态间互信息。

**CLIP 对比学习中的温度参数 $\tau$**

温度参数 $\tau$ 控制相似度分布的尖锐程度：
- $\tau \to 0$：相似度分布趋于 one-hot，梯度集中在最难负样本上（hard negative mining）
- $\tau \to \infty$：相似度分布趋于均匀，所有负样本梯度均等

CLIP 将 $\tau$ 设计为可学习参数（初始化为 $e^{0.07} \approx 1.07^{-1}$，即 $\tau = 1/e^{0.07}$），并加以 clamp 防止过小。训练收敛时 $\tau \approx 0.01 \sim 0.07$，模型自动找到最优的温度平衡点。

**对比学习的几何理解**

对比学习本质上在优化一个**超球面上的均匀分布约束**（von Mises-Fisher 分布视角）：
- L2 归一化将所有特征映射到单位超球面 $\mathbb{S}^{d-1}$
- 对比损失使配对的图文特征在超球面上靠近（内积大）
- 同时使非配对特征在超球面上均匀分散（避免表示坍缩）

这一几何结构解释了为什么 CLIP 特征在 zero-shot 迁移上表现优异：特征均匀分散在超球面上意味着决策边界在各方向都是清晰的，线性分类器（点积相似度）足以区分不同类别。

### 3.2 投影对齐（Projection-based Alignment）

对比学习仅对齐全局特征（图像和文本的 CLS token），而 MLLM 需要的是**细粒度的视觉特征与语言 token 的逐位对齐**。投影对齐方法直接将视觉特征投影到 LLM 的嵌入空间，作为 LLM 的输入 token。

#### 3.2.1 Q-Former（BLIP-2）：可学习查询向量与交叉注意力

Q-Former（Querying Transformer, Li et al., 2023）是 BLIP-2 提出的关键对齐模块，其设计动机是：直接将所有视觉 token（256 个）传给 LLM 计算成本过高，需要一个**信息压缩机制**将视觉特征压缩为固定数量的查询向量。

**Q-Former 的架构**

Q-Former 由两部分组成：
1. **图像 Transformer**：通过交叉注意力与冻结的视觉编码器特征交互
2. **文本 Transformer**：与文本 token 交互，共享参数

核心机制是一组**可学习查询向量** $Q = [q_1, q_2, \ldots, q_K] \in \mathbb{R}^{K \times d_Q}$（$K=32$），这些查询向量通过交叉注意力从视觉特征中提取信息：

$$\text{Cross-Attn}(Q, V_{\text{vis}}, V_{\text{vis}}) = \text{softmax}\left(\frac{Q W^Q_c \cdot (V_{\text{vis}} W^K_c)^\top}{\sqrt{d_k}}\right) \cdot V_{\text{vis}} W^V_c$$

其中 $V_{\text{vis}} \in \mathbb{R}^{N_{\text{vis}} \times d_{\text{vis}}}$ 为视觉编码器输出的特征序列。

**Q-Former 的三阶段训练**

```
阶段 1：视觉-语言表示学习（冻结视觉编码器）
├── Image-Text Contrastive Learning（ITC）
├── Image-Text Matching（ITM）
└── Image-grounded Text Generation（ITG）

阶段 2：生成式预训练（冻结视觉编码器 + Q-Former）
└── 将 Q-Former 输出线性投影至 LLM 嵌入空间，作为 soft visual prompt
```

**Q-Former 的数学意义**：$K$ 个查询向量的学习过程本质上是一个**视觉信息的有损压缩**——从 $N_{\text{vis}} \approx 256$ 个视觉 token 提取 $K=32$ 个最具代表性的信息摘要。这类似于注意力机制的 Attention Pooling，但增加了可学习的查询参数，允许模型动态决定"关注哪些视觉特征"。

#### 3.2.2 线性投影 vs MLP 投影 vs Q-Former 的权衡

LLaVA 系列的核心贡献之一是证明了**简单的 MLP 投影已足够**。设视觉特征 $z_{\text{vis}} \in \mathbb{R}^{N_{\text{vis}} \times d_{\text{vis}}}$，不同投影方式如下：

**线性投影（LLaVA v1.0）**：
$$h_i = z_i^{\text{vis}} W_{\text{proj}}, \quad W_{\text{proj}} \in \mathbb{R}^{d_{\text{vis}} \times d_{\text{llm}}}$$

**两层 MLP 投影（LLaVA v1.5）**：
$$h_i = W_2 \cdot \text{GELU}(W_1 z_i^{\text{vis}} + b_1) + b_2$$

**Q-Former 投影（BLIP-2）**：通过可学习查询和交叉注意力进行动态、上下文感知的压缩（参数量大，计算成本高）

| 投影方式 | 视觉 token 数量 | 参数量 | 计算量 | 动态性 |
|---------|--------------|--------|--------|--------|
| 线性投影 | 不变（$N_{\text{vis}}$） | $d_v \times d_l$ | 极低 | 无 |
| MLP 投影 | 不变（$N_{\text{vis}}$） | $2d_v d_l$ | 低 | 无 |
| Q-Former | 固定 $K=32$ | $\sim$188M | 中等 | 有（交叉注意力） |
| Perceiver Resampler | 固定 $K$ | $\sim$50M | 中等 | 有 |
| C-Abstractor（InternVL2） | $\lfloor N_{\text{vis}}/4 \rfloor$ | 轻量 | 低 | 有（2D卷积） |

**LLaVA 线性投影为何有效的深层原因**：LLM 的嵌入空间具有极强的泛化性和语义表达能力——预训练 LLM 已经在海量文本上学会了丰富的语义先验。一个简单的线性变换足以将视觉特征"翻译"到 LLM 能够理解的空间，前提是有足够多的视觉-语言配对数据来训练这个翻译器。这类似于词嵌入：尽管词嵌入的变换非常简单（查表），但 Transformer 本身的注意力机制足以在其上实现复杂的语义组合。

### 3.3 生成式对齐（Generative Alignment）

生成式对齐是最为直接的方法：将视觉 token 直接插入 LLM 的输入序列，与文本 token 拼接后通过自回归训练端到端优化。

**自回归损失**：给定图像 $I$ 和前缀文本 $x_{<t}$，LLM 的训练目标是：

$$\mathcal{L}_{\text{gen}} = -\sum_{t=1}^T \log p_\theta(x_t \mid f_{\text{vis}}(I), x_{<t})$$

其中 $f_{\text{vis}}(I) \in \mathbb{R}^{N_{\text{vis}} \times d_{\text{llm}}}$ 为视觉特征投影后的序列，**作为 LLM 输入的前缀（prefix）**。

**关键设计选择**：仅在文本部分计算损失（与 SFT 的 mask 机制一致），视觉 token 作为条件信息而非生成目标。

```python
# 生成式对齐的数据流（伪代码）
def mllm_forward(image, input_ids, labels):
    # 1. 视觉特征提取
    vis_features = vision_encoder(image)     # [B, N_vis, d_vis]
    
    # 2. 投影到 LLM 空间
    vis_tokens = projector(vis_features)      # [B, N_vis, d_llm]
    
    # 3. 文本 token 嵌入
    txt_embeds = llm.embed_tokens(input_ids)  # [B, L_txt, d_llm]
    
    # 4. 拼接视觉和文本 token
    # 格式：[BOS] <image tokens> [IMG_END] <question> [SEP] <answer>
    inputs_embeds = torch.cat([vis_tokens, txt_embeds], dim=1)
    
    # 5. LLM 前向传播
    logits = llm(inputs_embeds=inputs_embeds).logits
    
    # 6. 仅对答案部分计算损失（掩码问题和视觉部分）
    # labels 中图像位置和问题位置填 -100（忽略）
    loss = cross_entropy(logits[:, N_vis:], labels, ignore_index=-100)
    return loss
```

**图像 token 在 LLM 中的处理机制**：LLM 的因果注意力（Causal Attention）允许文本 token 双向注意视觉 token（视觉 token 在时间上早于文本），但视觉 token 之间不能相互注意文本 token（未来信息）。这意味着视觉特征作为"上下文前缀"被整个文本序列所利用，符合"先看图再回答"的认知顺序。

---

## 4. LLaVA 系列深度解析

LLaVA（Large Language and Vision Assistant）系列是 MLLM 领域最具影响力的开源工作之一，以其极简的架构设计和出色的性能著称。从 v1.0 到 OneVision，LLaVA 系列清晰地展示了 MLLM 架构演化的核心脉络。

### 4.1 LLaVA v1.0：极简主义的力量

**架构**：CLIP ViT-L/14 → 线性投影 → Vicuna-7B/13B

LLaVA v1.0（Liu et al., 2023）的架构极其简洁：冻结的 CLIP 视觉编码器提取图像特征，一个单层线性矩阵将视觉特征映射到 LLM 的嵌入维度，然后直接输入 Vicuna（基于 LLaMA-1 的 SFT 版本）。

**两阶段训练策略**

*阶段一：特征对齐预训练*（Pretraining for Feature Alignment）

- **数据**：CC-595K（LAION-CC-SBU 的子集，595K 图文对）
- **训练目标**：让线性投影矩阵学会将 CLIP 视觉特征映射到 Vicuna 的语义空间
- **冻结策略**：视觉编码器 ✓ 冻结，LLM ✓ 冻结，仅训练线性投影层
- **任务格式**：简单图像描述（"Describe the image briefly."）

*阶段二：端到端指令微调*（Fine-tuning End-to-End）

- **数据**：LLaVA-Instruct-158K（GPT-4 生成的多轮对话数据）
- **训练目标**：让整个系统学会遵循复杂的视觉问答指令
- **冻结策略**：视觉编码器 ✓ 冻结，LLM ✗ 不冻结，线性投影层 ✗ 不冻结

**GPT-4 指令数据生成**

LLaVA 的数据构造方法极具创新性。由于高质量的视觉指令数据稀缺，Liu 等人利用 GPT-4 的纯文本版本来生成训练数据：

1. 从 COCO 数据集获取图像的**详细文字描述**（captions）和**边界框标注**（bounding boxes）
2. 将这些文字信息作为上下文，让 GPT-4 生成基于图像的**对话数据**、**详细描述**和**复杂推理问题**
3. 利用 GPT-4 的语言推理能力，在没有真正"看到图像"的情况下，生成如同看过图像般的高质量问答

这一方法揭示了一个深刻洞察：**LLM 的语言理解能力可以通过文字代理（symbolic proxy）来补偿视觉感知的缺失**——只要文字描述足够准确，GPT-4 就能生成接近真实视觉理解的指令数据。

### 4.2 LLaVA v1.5：关键改进的量化分析

LLaVA v1.5（Liu et al., 2023b）在 v1.0 基础上做了三项关键改进，在 11 个基准上平均提升了约 7%：

**改进 1：CLIP-ViT-L-336px（高分辨率视觉编码器）**

将输入分辨率从 224×224 提升至 336×336，视觉 token 数量从 256 增加至 576。

$$\Delta N_{\text{vis}} = \left(\frac{336}{224}\right)^2 \times 256 = 2.25 \times 256 = 576$$

这一改进对需要细节理解的任务（OCR、图表识别）效果显著——更多的视觉 token 意味着每个 token 覆盖更小的图像区域（从 $14 \times 14$ 像素降至实际相同，但更高的输入分辨率保留了更多细节）。

**改进 2：MLP 投影层替代线性投影**

两层 MLP（带 GELU 激活）替代单层线性变换，在相同参数量下提供更强的非线性表达能力。消融实验显示，MLP 投影在视觉推理类任务上优于线性投影约 1-3%。

**改进 3：学术任务数据融合**

引入了 VQAv2、GQA、OKVQA、TextVQA、OCRVQA 等学术 VQA 数据集，让模型在对话指令微调的同时，也能学习细粒度的视觉问答。

**LLaVA v1.5 训练数据组成**：

| 数据集 | 类型 | 数量 | 阶段 |
|--------|------|------|------|
| LAION-CC-SBU-595K | 图文对 | 595K | 预训练 |
| LLaVA-Instruct-158K | GPT-4 指令 | 158K | 微调 |
| VQAv2 | 视觉问答 | 83K | 微调 |
| GQA | 结构化问答 | 72K | 微调 |
| TextVQA | OCR 问答 | 21K | 微调 |
| OCRVQA | OCR 问答 | 80K | 微调 |
| ShareGPT4V | 详细描述 | 100K | 微调 |

### 4.3 LLaVA v1.6 / LLaVA-NeXT：动态高分辨率突破

LLaVA-NeXT（Liu et al., 2024）解决了 v1.5 的核心局限——**固定低分辨率无法处理需要细节理解的场景**（文档、图表、屏幕截图等）。

**核心技术：AnyRes（任意分辨率处理）**

AnyRes 的思路类似于人类处理图像的方式：先整体浏览（thumbnail），再局部细看（patches）。

给定任意分辨率的输入图像 $I$，AnyRes 的处理流程：

1. **缩略图**：将图像 resize 到基础分辨率（336×336），提取全局上下文
2. **动态切片**：根据预设的分辨率网格（如 $\{672, 1344\} \times \{672, 1344\}$），选择最匹配输入纵横比的网格方案，将图像划分为 $n_r \times n_c$ 个子图（每个子图 336×336）

$$n_r \times n_c = \arg\min_{(r,c) \in \text{Grid}} \left[\text{aspect\_ratio\_diff}(r/c, H/W) + (r \times c - 1) \cdot \lambda\right]$$

3. **特征提取**：对每个子图分别通过视觉编码器提取特征，然后拼接
4. **视觉 token 序列**：$[Z_{\text{thumbnail}}; Z_{\text{patch}_{1,1}}; \ldots; Z_{\text{patch}_{n_r, n_c}}]$

**总视觉 token 数量**：

$$N_{\text{total}} = N_{\text{base}} + n_r \times n_c \times N_{\text{base}} = (1 + n_r \times n_c) \times 576$$

对于 4 个切片（2×2 网格），总视觉 token 数量为 $5 \times 576 = 2880$——这是 v1.5 的 5 倍，但有效分辨率从 336 提升到 672（细节识别能力显著提升）。

**AnyRes 与 token 行分隔符**：为了让 LLM 理解空间布局，LLaVA-NeXT 在每行切片后插入特殊 token `\n`，使 LLM 能感知图像的空间结构：

```
[BOS] [image_global: 576 tokens] 
[image_patch_row1_col1: 576 tokens] [image_patch_row1_col2: 576 tokens] \n
[image_patch_row2_col1: 576 tokens] [image_patch_row2_col2: 576 tokens] \n
[question tokens] [SEP] [answer tokens]
```

**LLaVA 系列演进总结**：

| 版本 | 视觉编码器 | 投影层 | LLM | 分辨率 | 视觉 Token |
|------|-----------|--------|-----|--------|-----------|
| v1.0 | CLIP ViT-L/14 | 线性 | Vicuna-7/13B | 224×224 | 256 |
| v1.5 | CLIP ViT-L/14@336px | MLP (2层) | Vicuna/LLaMA-2 7/13B | 336×336 | 576 |
| v1.6 | CLIP ViT-L/14@336px | MLP (2层) | Mistral-7B/Mixtral-8×7B/LLaMA-3 | up to 1344 | ≤2880 |
| OneVision | SigLIP-SO400M@384px | MLP (2层) | LLaMA-3-8B | 动态 | 动态 |

### 4.4 LLaVA-OneVision：统一单图、多图、视频

LLaVA-OneVision（Li et al., 2024）将 LLaVA 扩展到多图像和视频场景，核心创新是**统一的训练范式**：

$$\text{输入} = [\underbrace{I_1 \text{ tokens}, \ldots, I_n \text{ tokens}}_{\text{n 张图像或视频帧}}, \text{文本 tokens}]$$

视频理解通过将视频帧均匀采样为静态图像序列来处理，利用 LLM 的长序列建模能力来理解时序信息。这一设计的优雅之处在于：视频理解成为图像理解的自然扩展，无需单独设计时序模块。

---

## 5. Qwen-VL 系列深度解析

Qwen-VL 系列（Bai et al., 2023; Wang et al., 2024）是目前开源 MLLM 中综合性能最强的系列之一，在文档理解、精细粒度定位等任务上表现尤为突出。

### 5.1 Qwen-VL：基础架构

**架构组成**：
- **视觉编码器**：ViT（OpenCLIP ViT-bigG 初始化），参数量约 1.9B
- **视觉-语言适配器（VL Adapter）**：单层交叉注意力，将视觉特征压缩为 256 个 token
- **语言模型**：Qwen-7B

**位置感知的交叉注意力适配器**

Qwen-VL 的适配器与 Q-Former 类似，但更轻量：使用 $K=256$ 个可学习查询向量，通过单层交叉注意力从 ViT 的输出（1024 个 token，对于 224×224 输入）中提取信息，将视觉 token 数量从 $O(N_{\text{vis}})$ 固定压缩为 256。

**定位训练**：Qwen-VL 在训练数据中引入了大量包含边界框坐标的样本，使模型能够输出精确的位置信息：

```
问题：请找到图中所有的汽车并给出位置。
回答：<box>[(15,43),(312,287)]</box> 左侧轿车，<box>[(445,12),(622,198)]</box> 右侧SUV
```

坐标以相对坐标（0-1000 范围的整数）表示，通过特殊 token 包裹，使位置信息成为文本生成的一部分。

### 5.2 Qwen2-VL：动态分辨率与 M-RoPE

Qwen2-VL 是架构创新最为密集的版本，引入了两项关键技术：**原生动态分辨率（Naive Dynamic Resolution）** 和 **多模态旋转位置编码（M-RoPE）**。

#### 5.2.1 原生动态分辨率

与 LLaVA 的 AnyRes（切片 + 缩略图）不同，Qwen2-VL 直接处理原始分辨率图像，无需预设网格或切片：

1. 任意分辨率图像直接通过 ViT，按 $14 \times 14$ Patch 划分
2. 视觉 token 数量随图像大小动态变化（$N_{\text{vis}} = HW/196$）
3. 通过 **2×2 Merge**（类似于 PixelShuffle）将相邻 4 个 patch token 合并为 1 个，将视觉 token 数量减少到 $N_{\text{vis}} / 4$

**2×2 Merge 的数学操作**：

$$\tilde{z}_{i,j} = W_{\text{merge}} \cdot [z_{2i,2j}; z_{2i+1,2j}; z_{2i,2j+1}; z_{2i+1,2j+1}]$$

其中 $W_{\text{merge}} \in \mathbb{R}^{d_{\text{llm}} \times 4d_{\text{vis}}}$，将 4 个空间相邻 token 合并为 1 个更高维的 token。这本质上是**卷积的逆操作**（PixelUnshuffle / Space-to-Depth），在保留空间信息的前提下压缩 token 数量。

**动态分辨率的效果**：对于一张 1024×768 的图像：
- LLaVA-NeXT（4 切片）：$5 \times 576 = 2880$ token
- Qwen2-VL：$\frac{1024 \times 768}{14^2 \times 4} \approx \frac{786432}{784} \approx 1003$ token

Qwen2-VL 在更低 token 数量的情况下保留了更高的有效分辨率，因为每个最终 token 覆盖 $28 \times 28$ 像素（无信息损失），而切片方法需要 cross-patch 边界信息的拼接和可能的信息重复。

#### 5.2.2 M-RoPE：多模态旋转位置编码

传统 RoPE（Rotary Position Embedding，Su et al., 2022）为 1D 序列中的每个位置分配一个旋转角度：

$$\text{RoPE}(q, m) = q \cdot e^{im\theta}$$

其中 $m$ 为位置索引，$\theta$ 为频率基数。这对于文本 token 是自然的，但对于二维图像 token 存在根本性问题——**1D 的位置编码无法捕捉图像的 2D 空间结构**。

**M-RoPE 的设计**：Qwen2-VL 将 RoPE 扩展为多维形式，对视觉 token 分配三个独立的位置坐标 $(t, h, w)$：

- $t$：时间维度（图像取值为 0，视频帧取值为帧索引）
- $h$：行坐标（Patch 行索引）
- $w$：列坐标（Patch 列索引）

对于 $d$ 维的查询向量，M-RoPE 将其分为三份，分别用 $t$、$h$、$w$ 进行旋转：

$$\tilde{q}_d = q_d \cdot e^{i \cdot m_{(d)} \cdot \theta_d}$$

其中 $m_{(d)}$ 根据维度 $d$ 所属的坐标分量取对应的 $t/h/w$ 值：

$$m_{(d)} = \begin{cases} t & d \in [0, d/3) \\ h & d \in [d/3, 2d/3) \\ w & d \in [2d/3, d) \end{cases}$$

**对于文本 token**：$t = h = w = \text{sequence position}$，退化为标准 1D RoPE。

**M-RoPE 的关键性质**

1. **视觉 token 的空间关系被显式编码**：两个视觉 token 的相对旋转角度反映了它们在图像中的相对空间距离，而非仅仅是序列顺序
2. **图像-文本的注意力对齐更准确**：文本 token 可以通过位置旋转"寻址"到特定图像区域
3. **视频理解的自然扩展**：时间维度 $t$ 允许模型感知帧间时序关系，视频帧的 $h, w$ 坐标相同，仅 $t$ 不同，使模型能区分"同一位置在不同时刻"

```python
# M-RoPE 的简化实现
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, position_ids, mrope_section):
    # mrope_section: [d_t, d_h, d_w]，三个分量占用的维度数量
    # position_ids: [B, 3, L]，三个坐标的位置索引
    
    cos = cos[position_ids]  # [B, 3, L, d/2]
    sin = sin[position_ids]  # [B, 3, L, d/2]
    
    # 将维度分组，分别应用对应坐标的旋转
    cos_t, cos_h, cos_w = torch.split(cos, mrope_section, dim=1)
    sin_t, sin_h, sin_w = torch.split(sin, mrope_section, dim=1)
    
    cos_merged = torch.cat([cos_t[:, 0], cos_h[:, 1], cos_w[:, 2]], dim=-1)
    sin_merged = torch.cat([sin_t[:, 0], sin_h[:, 1], sin_w[:, 2]], dim=-1)
    
    # 标准 RoPE 旋转
    q_embed = (q * cos_merged) + (rotate_half(q) * sin_merged)
    k_embed = (k * cos_merged) + (rotate_half(k) * sin_merged)
    return q_embed, k_embed
```

### 5.3 Qwen2.5-VL 的进一步改进

Qwen2.5-VL 在 Qwen2-VL 基础上继续优化：

1. **Window Attention for ViT**：ViT 的自注意力在大分辨率下计算成本为 $O(N^2)$，Qwen2.5-VL 引入窗口注意力（将图像分为 $8 \times 8$ 的窗口，仅在窗口内计算注意力），将视觉编码器的复杂度从 $O(N^2)$ 降至 $O(N)$。每 4 层插入一次全局注意力以捕捉长程依赖。

2. **更精细的视频时序理解**：通过 M-RoPE 的时间维度，模型能感知具体的时间戳（以秒为单位），不再仅是帧索引，使时序推理更准确。

3. **增强的文档理解**：训练数据中大幅增加高质量的文档、图表、代码截图数据，在 DocVQA、ChartQA 等任务上取得显著提升。

---

## 6. 其他重要模型

### 6.1 BLIP-2 与 InstructBLIP

**BLIP-2**（Li et al., 2023）是 Q-Former 架构的集大成者。其最大贡献是通过 Q-Former 将**任意预训练视觉编码器**与**任意预训练 LLM** 解耦，使视觉编码器的升级（如从 ViT-L 到 EVA-ViT-G）和 LLM 的替换（如从 OPT 到 FlanT5 到 Vicuna）成为即插即用的操作。

BLIP-2 的训练分为两个核心阶段：

*阶段一*（视觉-语言表示学习）使用三个损失联合优化 Q-Former：

$$\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{ITC}} + \mathcal{L}_{\text{ITM}} + \mathcal{L}_{\text{ITG}}$$

- $\mathcal{L}_{\text{ITC}}$（Image-Text Contrastive）：对齐 Q-Former 输出与文本特征的全局表示
- $\mathcal{L}_{\text{ITM}}$（Image-Text Matching）：二分类任务，判断图文是否匹配
- $\mathcal{L}_{\text{ITG}}$（Image-grounded Text Generation）：以图像为条件生成文本

**InstructBLIP**（Dai et al., 2023）在 BLIP-2 基础上引入了**指令感知的视觉特征提取**：在 Q-Former 的交叉注意力中引入指令文本作为额外条件，使 Q-Former 提取的视觉特征与当前指令更相关。

$$Q_{\text{instruct}} = \text{CrossAttn}(Q, V_{\text{vis}}, V_{\text{vis}} | T_{\text{instruct}})$$

数学上，指令文本通过自注意力影响查询向量 $Q$ 的状态，再通过交叉注意力从视觉特征中提取指令相关的信息。这使得同一张图像在不同问题下能提取到不同的"关键信息"——问"图中有几辆车？"时，Q-Former 应关注检测相关特征；问"图中的天空是什么颜色？"时，应关注颜色和区域特征。

### 6.2 InternVL 系列

InternVL（Chen et al., 2024）的核心特色是使用了**超大规模的视觉编码器**（InternViT-6B），打破了"视觉编码器远小于 LLM"的惯例。

**核心洞察**：MLLM 的瓶颈往往在视觉理解能力，而非语言生成能力。将视觉编码器规模从 307M（CLIP ViT-L）扩展到 6B，可以大幅提升视觉特征的质量，即使语言模型相对较小（如 InternLM-7B），总体性能也能超越更大的 LLM + 小视觉编码器的组合。

**InternVL2 的连接器设计（C-Abstractor）**：

```
视觉 Tokens [B, N, d_vis]
    ↓ Pixel Shuffle（Space-to-Depth）
压缩 Tokens [B, N/4, 4*d_vis]
    ↓ MLP 映射
LLM Tokens [B, N/4, d_llm]
```

C-Abstractor 通过 2D 卷积的降采样（等效于 Pixel Shuffle）同时压缩 token 数量和调整特征维度，比 Q-Former 参数更少、速度更快，同时保留了比线性投影更强的空间感知能力。

### 6.3 MiniCPM-V：超高效设计

MiniCPM-V（Yu et al., 2024）专注于轻量化 MLLM 的端侧部署，在极小的参数量（2B-8B）下实现接近 GPT-4V 的性能。

**Token 压缩策略**：MiniCPM-V 引入了自适应视觉编码（Adaptive Visual Encoding），根据图像内容动态决定视觉 token 数量，避免简单图像消耗过多 token：

- 对于简单图像（如纯色背景）：使用少量 token
- 对于复杂图像（如密集文字文档）：自适应增加 token 数量

**实现机制**：通过预测每个区域的"重要性分数"（importance score），使用动态阈值过滤低重要性的视觉 token，类似于注意力机制的稀疏化版本。

### 6.4 DeepSeek-VL2

DeepSeek-VL2 将 DeepSeek 的 MoE 架构延伸至多模态领域，其核心创新是**视觉 token 感知的路由策略**——视觉 token 和文本 token 的路由分布天然不同，通过在路由器中引入模态感知偏置（modal-aware bias），使视觉和文本 token 倾向于被不同的专家处理，实现视觉和语言知识的自然解耦：

$$\text{gate}(x, m) = \text{softmax}((W_r \cdot x + b_m) / T)$$

其中 $m \in \{\text{visual}, \text{text}\}$ 为模态标识，$b_m$ 为模态特定的偏置项。

---

## 7. 连接器/投影器设计（Connector Design）

连接器（Connector）是 MLLM 中桥接视觉编码器和 LLM 的关键模块，其设计直接影响模型的效率和性能。

### 7.1 连接器设计空间的系统分析

设视觉编码器输出 $V \in \mathbb{R}^{N_v \times d_v}$，LLM 期望输入 $H \in \mathbb{R}^{N_h \times d_h}$，连接器需要解决：

1. **维度对齐**：$d_v \neq d_h$（如 ViT-L 的 1024 维 vs LLaMA 的 4096 维）
2. **序列长度压缩**：$N_v$ 往往过多（如 576），需要压缩到合理数量
3. **特征质量**：从视觉特征中提取与任务相关的信息

不同连接器的形式化比较：

**线性投影**：
$$H = V W, \quad W \in \mathbb{R}^{d_v \times d_h}, \quad N_h = N_v$$

**MLP 投影**：
$$H = W_2 \sigma(W_1 V + b_1) + b_2, \quad N_h = N_v$$

**Q-Former**：
$$H = \text{CrossAttn}(Q, V, V), \quad Q \in \mathbb{R}^{K \times d_q}, \quad N_h = K \ll N_v$$

**Perceiver Resampler**（Flamingo, Alayrac et al., 2022）：
$$H = \text{CrossAttn}(Q_{\text{learned}}, V, V), \quad \text{多层堆叠}$$

**Pixel Shuffle（PixelUnshuffle）**：
$$H[i,j] = [V[2i, 2j]; V[2i+1, 2j]; V[2i, 2j+1]; V[2i+1, 2j+1]] W, \quad N_h = N_v/4$$

**Token Merging（ToMe）**：
通过计算相邻 token 的余弦相似度，合并最相似的 token 对：
$$\text{ToMe}: \text{merge}(v_i, v_j) = \frac{s_i \cdot v_i + s_j \cdot v_j}{s_i + s_j}, \quad \text{if similarity}(v_i, v_j) > \tau$$

### 7.2 视觉 Token 数量的效率-精度权衡

视觉 token 数量 $N_h$ 的选择是 MLLM 设计中最重要的超参数之一，直接影响：

1. **LLM 推理成本**：注意力计算 $O((N_h + L_t)^2)$，$N_h$ 增大导致二次增长
2. **KV Cache 显存**：每层存储 $N_h \times d_h$ 的 KV 缓存
3. **视觉信息保留量**：$N_h$ 过小时，视觉细节（OCR、细粒度定位）丢失

**实验数据**（来自 InternVL2 和 MiniCPM-V 的消融实验）：

| 视觉 Token 数 | TextVQA (↑) | DocVQA (↑) | MMBench (↑) | 推理延迟（相对值） |
|-------------|------------|-----------|-----------|----------------|
| 64 | 62.1 | 55.3 | 74.2 | 1.0× |
| 256 | 72.4 | 68.1 | 78.9 | 1.8× |
| 576 | 74.1 | 72.5 | 80.3 | 2.9× |
| 1024 | 75.2 | 76.8 | 81.1 | 4.7× |
| 2304 | 75.8 | 80.2 | 81.5 | 9.3× |

**关键观察**：从 256 到 576 token 有显著的性能提升，但从 576 到 2304 的提升趋于边际，而延迟增长远超性能提升。这表明存在一个"效率膝点"（efficiency knee），约在 256-576 token 范围内。

对于需要 OCR 和文档理解的任务，高 token 数量的收益更大（DocVQA 从 256 到 2304 提升 12.1 个点），而对于一般性问答（MMBench），256 token 已经足够。

### 7.3 动态 Token 压缩策略

一个重要的研究方向是根据图像内容动态调整视觉 token 数量，类似于 NLP 中的自适应计算（Adaptive Computation）：

**方案 1：重要性过滤**（Importance-based Filtering）

计算每个视觉 token 对 CLS token 的注意力权重，保留权重最高的 top-$k$ 个 token：

$$I_i = \frac{1}{H}\sum_{h=1}^H \text{Attn}^h_{\text{CLS} \to i}, \quad S_k = \{i : I_i \in \text{top-k}\}$$

**方案 2：相似性合并**（Similarity-based Merging）

在视觉编码器的某一层，将余弦相似度超过阈值的相邻 token 合并为加权平均：

$$v_{\text{merged}} = \sum_{i \in \text{cluster}} w_i v_i, \quad w_i \propto \exp(\text{importance}_i)$$

**方案 3：分辨率感知路由**（Resolution-aware Routing）

根据输入图像的实际内容复杂度（通过低分辨率预处理估计），动态选择"高分辨率（多 token）"或"低分辨率（少 token）"处理路径：

$$k^* = \arg\min_k \left[|k - \hat{k}_{\text{pred}}|^2 + \lambda \cdot k^2\right]$$

其中 $\hat{k}_{\text{pred}}$ 是基于图像复杂度的预测最优 token 数量，$\lambda$ 为效率惩罚系数。

---

## 8. 训练策略

MLLM 的训练是多阶段的精细工程，不同阶段的目标、数据和冻结策略各有不同。

### 8.1 两阶段训练（LLaVA 范式）

**阶段一：视觉-语言特征对齐**（Feature Alignment）

目标：让连接器学会将视觉特征"翻译"到 LLM 的语义空间。

- **数据**：大规模图文对（CC3M、LAION 等，通常 500K-1M 级别），任务为简单的图像描述
- **冻结策略**：视觉编码器 ✓ 冻结，LLM ✓ 冻结，**仅训练连接器**
- **数学目标**：最小化连接器参数 $\Phi_c$ 上的生成损失：

$$\Phi_c^* = \arg\min_{\Phi_c} \sum_{(I, T) \in \mathcal{D}_1} \mathcal{L}_{\text{gen}}(f_c(f_v(I)), T; \theta_{\text{LLM}})$$

- **直觉**：此阶段不需要 LLM 学习新能力，只需连接器成为视觉和语言之间的"翻译桥梁"

**阶段二：指令跟随微调**（Instruction Tuning）

目标：让整个系统学会遵循多样化的视觉指令。

- **数据**：多样化的视觉问答数据、描述数据、推理数据（通常混合多个数据集）
- **冻结策略**：视觉编码器 ✓/✗（模型依赖），LLM ✗ 解冻，连接器 ✗ 解冻
- **数据格式**：多轮对话格式，仅对助手回答部分计算损失

**关键问题：视觉编码器应该冻结吗？**

这是 MLLM 训练中有争议的设计选择：

| 策略 | 优点 | 缺点 |
|------|------|------|
| **冻结视觉编码器** | 训练稳定，防止遗忘 CLIP 对齐的视觉表示；计算效率高（无需视觉编码器梯度） | 视觉特征固定，无法适应下游任务的特定需求 |
| **解冻视觉编码器** | 视觉特征可以根据语言指令自适应调整；细粒度任务（OCR、定位）收益明显 | 灾难性遗忘风险（可能破坏 CLIP 对齐），训练不稳定，成本高 |
| **低秩微调（LoRA）** | 在低秩空间内微调，保留大部分预训练特征 | LoRA 秩的选择影响性能 |

实践中，LLaVA 系列冻结视觉编码器，Qwen-VL 和 InternVL 在第二阶段解冻视觉编码器（使用更小的学习率），后者在细粒度任务上有更好表现。

### 8.2 三阶段训练

更精细的模型（如 InternVL2）采用三阶段训练：

```
阶段 0：视觉编码器预训练
    └── 用大规模图文对训练视觉编码器（CLIP 或 EVA 风格）

阶段 1：视觉-语言对齐
    └── 冻结视觉编码器，训练连接器（同两阶段范式的阶段一）

阶段 2：联合微调
    └── 解冻所有模块，用高质量多任务数据微调（通常配合 LoRA 防止过拟合）
```

### 8.3 多模态 SFT 数据构造方法

高质量的 SFT 数据是 MLLM 能力的关键来源。主要构造方法包括：

**方法 1：GPT-4V 批注**（GPT-4V Annotation）

使用 GPT-4V 对图像进行详细标注，生成高质量的描述、问答对和推理链：

```python
# GPT-4V 批注示意
prompt = """
你是一个专业的图像分析助手。请对以下图像进行详细分析：
1. 详细描述图像中的所有元素（物体、颜色、空间关系）
2. 提出3个关于图像的问题并给出答案
3. 分析图像中可能涉及的知识（科学、文化、地理等）
"""
response = gpt4v_api.call(image=image, prompt=prompt)
```

**方法 2：规则化数据增强**（Rule-based Augmentation）

从现有标注（COCO caption, VQA annotations, OCR labels 等）出发，用规则模板生成多样化的问答对：

```python
templates = {
    "counting": "图中有多少{object}？",
    "color": "{object}是什么颜色的？", 
    "position": "{object1}在{object2}的{direction}方向吗？",
    "OCR": "图中文字'{text}'表达了什么意思？"
}
```

**方法 3：合成数据管道**（Synthetic Data Pipeline）

对于专业任务（文档理解、图表问答），通过程序化生成图像-问题-答案三元组：

```python
# 图表数据合成示意
def generate_chart_qa(data, chart_type="bar"):
    # 1. 用 matplotlib 生成图表
    fig = create_chart(data, chart_type)
    image = render_to_image(fig)
    
    # 2. 基于数据生成问答
    questions = [
        f"哪个类别的值最高？答案：{argmax(data)}",
        f"{category_A} 的值是多少？答案：{data[category_A]}",
        f"{category_A} 比 {category_B} 高多少？答案：{data[A] - data[B]}"
    ]
    return image, questions
```

**方法 4：长链推理数据（CoT 扩展）**

将简单问答扩展为包含视觉推理链的格式，训练模型"先分析图像，再给出答案"：

```
输入：[图像] 这道数学题的答案是什么？

输出：让我仔细分析图像...
首先，图中有一个三角形，标注了两条边分别为 3cm 和 4cm，夹角为 90°。
根据勾股定理：c² = a² + b² = 9 + 16 = 25，所以 c = 5cm。
答案是：斜边长度为 5cm。
```

---

## 9. 评测基准

### 9.1 传统视觉问答与描述

**VQAv2**（Visual Question Answering v2）：图像-问题-答案三元组，问题涵盖物体识别、计数、颜色判断等基础能力。v2 版本通过引入"对立样本对"（两张相似图像，相同问题有不同答案）来减少语言偏置。

**NoCaps**（Novel Object Captioning）：测试模型对训练时未见过的物体类别的描述能力，评估视觉编码器的 zero-shot 泛化能力。

### 9.2 综合评测基准

现代 MLLM 通常用综合基准评估多种能力：

**MMBench**（Sun et al., 2023）：覆盖 20 个维度（感知、推理、知识等）的综合测评，通过选择题格式（CircularEval）消除输出格式的影响。

**MME**（Fu et al., 2023）：二进制是/否问答，在 14 个感知任务和 4 个认知任务上评测，提供细粒度的能力分析。

**SEED-Bench**（Li et al., 2023）：19K 个多项选择题，涵盖图像和视频理解的 12 个维度。

**MM-Vet**（Yu et al., 2023）：开放式问题，评测 6 种核心 VL 能力及其组合：识别（Recognition）、OCR、知识（Knowledge）、生成（Generation）、空间（Spatial）、数学（Math）。

$$\text{MM-Vet Score} = \sum_{c \in \text{capabilities}} w_c \cdot \text{GPT-eval}(c)$$

采用 GPT-4 作为自动评分器，赋予 0/1/2 分（完全错误/部分正确/完全正确）。

**综合评测结果对比**（截至 2024 年）：

| 模型 | MMBench | MME | SEED | MM-Vet | TextVQA |
|------|--------|-----|------|--------|---------|
| LLaVA-1.5-7B | 67.0 | 1510 | 65.7 | 30.5 | 58.2 |
| LLaVA-1.5-13B | 68.2 | 1531 | 68.2 | 35.4 | 61.3 |
| LLaVA-NeXT-7B | 69.6 | 1604 | 70.2 | 43.9 | 64.9 |
| Qwen-VL-Chat | 60.6 | 1487 | 65.4 | - | 61.5 |
| Qwen2-VL-7B | 80.5 | - | - | 60.6 | 84.3 |
| InternVL2-8B | 79.4 | - | - | 54.2 | 77.4 |
| InternVL2-76B | 86.5 | - | - | 68.4 | 84.4 |

### 9.3 文档与图表理解基准

文档理解是 MLLM 的重要应用场景，专用基准更能反映实际能力：

**DocVQA**：真实文档（发票、报告、表格）的问答，要求精确的 OCR 和文档布局理解。评测指标为 ANLS（Average Normalized Levenshtein Similarity），对比模型输出与标准答案的字符级相似度：

$$\text{ANLS} = \frac{1}{N} \sum_{i=1}^N \max\left(0, 1 - \frac{d(a_i, \hat{a}_i)}{\max(|a_i|, |\hat{a}_i|)}\right)$$

其中 $d(\cdot, \cdot)$ 为 Levenshtein 距离。

**ChartQA**：图表（柱状图、折线图、饼图）理解，同时包含直接读取数值（Extractive QA）和需要数学推理（Reasoning QA）两类问题。

**InfoVQA**：信息图（Infographic）的问答，包含文字、图标、数字等混合视觉元素。

**OCRBench**：专注于 OCR 能力的综合测评，涵盖场景文字识别、文档 OCR、手写文字等多种形式。

| 模型 | DocVQA | ChartQA | InfoVQA | TextVQA |
|------|--------|---------|---------|---------|
| LLaVA-1.5-7B | 28.1 | 18.2 | 25.8 | 58.2 |
| Qwen-VL-Chat | 62.6 | 65.7 | 44.8 | 61.5 |
| Qwen2-VL-7B | 94.5 | 83.0 | 76.5 | 84.3 |
| InternVL2-8B | 91.6 | 83.3 | 74.8 | 77.4 |
| GPT-4V | 87.2 | 78.5 | 75.1 | 78.0 |

Qwen2-VL 在文档理解上的突出表现（DocVQA 94.5）得益于其动态高分辨率处理和大量文档训练数据，已超过 GPT-4V。

### 9.4 视频理解

将 MLLM 扩展到视频理解是重要趋势：

**Video-Bench**：多选题形式的视频问答，涵盖动作识别、时序推理、视频描述等。

**EgoSchema**：第一人称视频的长视频理解（3 分钟），需要推理连续活动序列中的因果关系。

主要挑战在于视频的**时序冗余性**——相邻帧高度相似，如何高效采样而不丢失关键信息是核心问题：

- **均匀采样**：等间隔提取帧（如每秒 1 帧），简单但可能错过快速事件
- **关键帧提取**：基于光流或帧差异选择高运动帧
- **自适应采样**：根据查询问题动态决定采样密度（如问"视频中发生了什么意外"时，关注运动突变帧）

---

## 10. 开放挑战与未来方向

### 10.1 高分辨率与长序列的显存矛盾

高分辨率是提升 MLLM 性能（特别是 OCR、文档理解）的关键，但带来了严峻的显存挑战：

**KV Cache 显存分析**：

对于一个 $L$ 层、$d$ 维、$N_{\text{vis}} + N_{\text{txt}}$ 个 token 的 Transformer：

$$\text{KV Cache Size} = 2 \times L \times (N_{\text{vis}} + N_{\text{txt}}) \times d \times \text{dtype\_bytes}$$

以 LLaMA-3-8B（$L=32$, $d=4096$, bfloat16） + AnyRes（$N_{\text{vis}}=2880$）为例：

$$\text{KV Cache} = 2 \times 32 \times (2880 + 512) \times 4096 \times 2 \text{ bytes} \approx 3.4 \text{ GB}$$

这仅是单样本的 KV Cache，在 batch 推理时成倍增长。

**解决方案方向**：

1. **视觉 token 早期退出**（Visual Token Early Exit）：在 LLM 的前几层处理完视觉 token 后，将其从 KV Cache 中剔除（类似于 "Efficient MLLM" 研究中的 token dropping），后续层仅保留文本 token 的 KV Cache：

$$\text{Attn}^{(l)}(T, V) = \begin{cases} \text{Full Attention}(T \cup V) & l \leq L_{\text{drop}} \\ \text{Text-only Attention}(T) & l > L_{\text{drop}} \end{cases}$$

2. **Flash Attention + Ring Attention**：通过分块注意力计算（FlashAttention-2/3）和环形通信（Ring Attention），将显存从 $O(N^2)$ 降至 $O(N)$，支持超长序列

3. **视觉 token 量化**：将视觉 token 的 KV Cache 量化到 INT4/INT8，在精度几乎不损失的情况下降低 50%-75% 的显存占用

### 10.2 视频理解的时序建模困境

视频理解是 MLLM 的下一个主战场，但面临特殊挑战：

**时序冗余性**：视频中相邻帧内容高度相似，若直接将所有帧的视觉 token 拼接，会引入大量冗余信息，浪费计算资源。

**时序依赖性**：理解动作、事件需要建模帧间的时序关系，而静态图像的并行处理无法捕捉这种关系。

**现有方案的局限**：

- LLaVA-OneVision：视频帧当作独立图像处理，时序关系完全依赖 LLM 的位置编码
- VideoLLaMA2：引入时空连接器（Video Q-Former），但参数量大、训练复杂
- Qwen2-VL M-RoPE：时间维度的位置编码提供时序感知，但缺乏显式的时序聚合机制

**潜在方向**：将 3D 卷积或时序 Transformer 集成到视觉编码器中，在特征提取阶段就建模时序关系；或者借鉴视频领域的分层特征（动作级别 > 事件级别 > 场景级别）来设计多粒度视频 token。

### 10.3 "视觉 Token 算力浪费"问题

大量研究（如 Visual Redundancy in LLaVA）发现，MLLM 实际上并没有充分利用所有视觉 token——通过可视化 LLM 各层的注意力权重，发现大多数文本 token 主要关注少数"关键"视觉 token，而大量视觉 token 的注意力权重几乎为零。

**量化分析**：在 LLaVA-1.5 的 VQAv2 任务上，通过留一法（leave-one-out）评估每个视觉 token 的贡献度，发现：
- Top-10% 的视觉 token 贡献了约 60% 的注意力分数
- Bottom-50% 的视觉 token 贡献不足 5%

这意味着至少 50% 的视觉 token 是冗余的，在推理阶段可以安全剔除，对精度影响极小，却能将推理速度提升 2-3 倍。

**LLaVA-Prune / FastV** 等工作通过在 LLM 的第 $K$ 层后剔除低注意力视觉 token，实现了显著的推理加速：

```python
def fast_v_forward(self, inputs_embeds, attention_mask, visual_mask, prune_layer=2, prune_ratio=0.5):
    for layer_idx, layer in enumerate(self.layers):
        hidden_states = layer(hidden_states, attention_mask)
        
        if layer_idx == prune_layer:
            # 计算视觉 token 的平均注意力权重
            vis_attn = layer.self_attn.attn_weights[:, :, :, visual_mask]  # 文本→视觉的注意力
            vis_importance = vis_attn.mean(dim=[0, 1, 2])  # 平均值作为重要性
            
            # 保留重要性最高的 (1-prune_ratio) 个视觉 token
            k = int(vis_importance.shape[0] * (1 - prune_ratio))
            keep_idx = vis_importance.topk(k).indices
            
            # 从 hidden_states 中移除低重要性视觉 token
            hidden_states = drop_visual_tokens(hidden_states, visual_mask, keep_idx)
            attention_mask = update_attention_mask(attention_mask, visual_mask, keep_idx)
    
    return hidden_states
```

### 10.4 原生多模态（Early Fusion）的潜力与挑战

当前对齐融合（Late Fusion）范式的根本局限在于：视觉信息仅通过连接器的一次性投影进入 LLM，LLM 各层内部无法再与原始视觉特征交互。这可能限制了需要多轮视觉-文本交互的复杂推理任务。

**Early Fusion 的架构设想**：

- 不区分"视觉编码器"和"语言模型"，所有模态的 token 在同一个 Transformer 中联合处理
- 每一层的 self-attention 同时覆盖视觉 token 和文本 token
- 视觉和语言表示在每一层都相互影响

**Chameleon（Meta, 2024）** 是早期 Early Fusion 尝试之一：将图像离散化为 512×512 的 token 序列（通过 VQ-VAE），与文本 token 统一训练。但发现训练稳定性差——视觉 token 和文本 token 的分布差异导致梯度冲突，需要精心的架构设计（如视觉和文本使用独立的归一化层）。

**未来方向猜想**：

$$\text{理想架构} = \underbrace{\text{Sparse Mixture of Experts}}_{\text{视觉/文本专家分离}} + \underbrace{\text{Early Fusion Attention}}_{\text{深层跨模态交互}} + \underbrace{\text{Dynamic Resolution}}_{\text{自适应视觉精度}}$$

通过 MoE 的路由机制自然实现视觉和文本专家的分离（参见 [DeepSeek-VL2 的视觉感知路由](#64-deepseek-vl2)），同时保留 Early Fusion 的深层跨模态交互能力。

### 10.5 多模态推理与 MLLM + RL

随着 R1 风格的推理训练（见 LLM 博客中 GRPO 章节）在纯文本 LLM 上取得巨大成功，将这一思路扩展到多模态领域成为重要研究方向。

**视觉推理的独特挑战**：

1. **奖励信号稀疏**：视觉数学题（如几何证明）的正确答案难以自动验证，需要更强的奖励模型
2. **视觉幻觉（Hallucination）**：模型可能"凭空想象"图中不存在的内容，RL 的探索过程可能加剧幻觉
3. **多步视觉推理**：需要反复"查看"图像的不同区域（类似人类的注视转移），当前 MLLM 缺乏这种动态视觉注意机制

**LLaVA-R1 / VisualR1 等工作**（2025）尝试将 GRPO 应用于多模态：

$$\mathcal{L}_{\text{GRPO-VL}} = -\mathbb{E}_{(I, q, a) \sim \mathcal{D}} \left[\mathbb{E}_{\{o_i\}_{i=1}^G \sim \pi_\theta(\cdot|I,q)} \sum_{i=1}^G \hat{A}_i \log \pi_\theta(o_i | I, q)\right]$$

其中 $\hat{A}_i$ 为相对于组内平均的归一化优势，通过格式奖励（CoT 格式正确）+ 答案奖励（数值正确）的混合奖励信号训练。

初步结果显示，RL 训练能够让 MLLM 学会"分步骤分析图像"的推理模式——模型自发地产生"首先观察图像中的...，然后注意到...，最终推理出..."的思维链，与 R1 在文本推理中发现的 "Aha Moment" 有相似的行为模式。

---

## 总结：MLLM 的技术图谱

本文从数学原理出发，系统梳理了多模态大模型的核心技术体系：

**从下到上的技术栈**：

```
┌──────────────────────────────────────────┐
│           应用层：评测 & 任务            │
│   VQA / OCR / DocVQA / Video Understanding│
├──────────────────────────────────────────┤
│           训练层：数据 & 策略            │
│   两阶段训练 / SFT数据构造 / RL扩展      │
├──────────────────────────────────────────┤
│           对齐层：连接器设计             │
│   线性/MLP投影 ←→ Q-Former ←→ Pixel Shuffle│
├──────────────────────────────────────────┤
│           对齐层：图文对齐               │
│   对比学习(CLIP) ←→ 生成式对齐(LLaVA)  │
├──────────────────────────────────────────┤
│           感知层：视觉编码器             │
│   ViT-L/14 → SigLIP → EVA-CLIP-18B     │
└──────────────────────────────────────────┘
```

**当前最佳实践的主流选择**：

| 组件 | 2023 年标配 | 2024-2025 最佳实践 |
|------|-----------|------------------|
| 视觉编码器 | CLIP ViT-L/14 (307M) | SigLIP-400M 或 InternViT-6B |
| 连接器 | 线性投影 | MLP 或 Pixel Shuffle |
| 分辨率策略 | 固定 224/336 | 动态分辨率（AnyRes 或原生动态） |
| 位置编码 | 标准 1D RoPE | M-RoPE（2D/3D 感知） |
| LLM | LLaMA-2 7B/13B | LLaMA-3 / Qwen2.5 8B-72B |
| 训练阶段 | 2阶段 | 2-3阶段，视觉编码器可选解冻 |

**未来最值得关注的方向**：

1. **视觉 token 效率**：动态 token 压缩、FastV 风格的运行时剪枝——实现高分辨率与低延迟的同时满足
2. **多模态 RL**：将 R1/GRPO 思路扩展到视觉推理，探索"视觉思维链"的涌现机制
3. **Early Fusion 原生多模态**：彻底消除"视觉编码器 + LLM"的二元架构，在统一的 Transformer 中实现更深层的视觉-语言交互
4. **视频理解**：时序感知的视觉编码、高效的视频 token 压缩、长视频的分层理解

从 CLIP 的对比学习到 Qwen2-VL 的 M-RoPE，从 LLaVA 的极简投影到 InternVL 的超大视觉编码器，多模态大模型正以令人惊叹的速度演化。视觉与语言的融合不仅是技术创新，更是向"具身智能"迈进的必经之路——真正理解世界的 AI，需要同时理解世界的样子和世界的语言。
