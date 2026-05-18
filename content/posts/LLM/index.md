---
date: '2026-04-19T16:36:00+08:00'
draft: false
title: 'LLM基础知识'
---
*由于最近在准备LLM相关的实习，但是在面试过程中发现对于一些理论还是不太熟悉，导致经常卡壳，所在就在这里记录一下这段时间学到的关于LLM的知识，也是帮助自己巩固一下。*

## Scaling Laws（缩放定律）

Scaling Laws 是理解 LLM 训练经济学和模型能力边界的理论基石。它回答了一个根本性问题：**在给定计算预算 $C$ 下，如何在模型参数量 $N$、训练数据量 $D$ 和训练时长之间做最优分配？** 这些定律不仅决定了预训练的资源配置策略，还深刻影响着 SFT、RL 和蒸馏等下游训练环节的工程决策。

### Kaplan Scaling Law（OpenAI, 2020）

Kaplan 等人通过对不同规模 Transformer 模型的系统性实验，首次揭示了测试损失与模型规模、数据量、计算量之间的**幂律关系**（power-law）。核心公式为：

$$L(N, D, C) = \left(\frac{N\_c}{N}\right)^{\alpha\_N} + \left(\frac{D\_c}{D}\right)^{\alpha\_D} + \left(\frac{C\_c}{C}\right)^{\alpha\_C} + L\_\infty$$

其中：
- $N$ 为非嵌入层参数量，$D$ 为训练数据量（tokens），$C \approx 6ND$ 为总计算量（PF-days）
- 实测指数：$\alpha\_N \approx 0.076$，$\alpha\_D \approx 0.095$，$\alpha\_C \approx 0.050$
- $L\_\infty$ 为理论上不可约减的最低损失（irreducible loss），由数据中固有的不可预测性决定

**关键结论："优先扩大模型规模"**。Kaplan 发现在给定计算预算 $C$ 下，增大模型参数量 $N$ 带来的损失降低幅度（指数 $\alpha\_N \approx 0.076$）显著高于单纯增加数据量 $D$（指数 $\alpha\_D \approx 0.095$，但数据传输瓶颈使得 $N$ 的边际收益更高）。他们给出的指导公式为：

$$N\_{\text{opt}} \propto C^{0.73},\quad D\_{\text{opt}} \propto C^{0.27}$$

即当计算预算翻倍时，应优先将大部分新增算力分配给模型规模的扩大，而非数据量的增加。

**为什么损失与规模服从幂律？** 这一关系的深层原因来自 SGD 优化过程中**梯度估计的方差**。LLM 训练中，每个 batch 的梯度是对全数据分布梯度的噪声估计：

$$\hat{g} = \frac{1}{B}\sum\_{i=1}^{B} \nabla\_\theta \log p\_\theta(y\_i|x\_i),\quad \mathbb{E}[\hat{g}] = g\_{\text{true}},\quad \text{Var}(\hat{g}) \propto \frac{1}{B}$$

噪声梯度的方差随着模型规模 $N$ 增大而累积，导致优化路径偏离最优点。增加数据量 $D$ 降低了每个 token 的过拟合风险，增加模型规模 $N$ 则提供了更多参数容量来捕捉数据的长尾分布结构。两者对损失的贡献以幂律衰减——这是在大规模随机优化中，**经验风险最小化收敛速度**的统计学习理论直接推论。

| 维度 | 参数 $N$ 的贡献 | 数据 $D$ 的贡献 |
|------|----------------|----------------|
| 内在机制 | 增大假设空间容量 | 降低泛化误差 |
| 幂律指数（Kaplan） | $\alpha\_N \approx 0.076$ | $\alpha\_D \approx 0.095$ |
| 主导因素 | 模型能否"学会" | 模型学到的东西"多准" |
| 边际收益递减速度 | 慢（Kaplan认为应优先投入） | 快（数据墙更早出现） |

### Chinchilla Scaling Law（DeepMind, 2022）

Hoffmann 等人的 Chinchilla 工作在 Scaling Laws 领域产生了**范式级修正**。他们指出，Kaplan 的分析存在一个关键缺陷：在扩大模型规模时，Kaplan 没有同步增加训练步数或调整学习率 schedule，导致大模型实际上**训练不充分（undertrained）**。结果严重低估了数据量的重要性。

Chinchilla 的重新分析给出了截然不同的结论：**在给定计算预算 $C$ 下，模型参数量 $N$ 和训练数据量 $D$ 应该同步等比例增长**：

$$N\_{\text{opt}} \propto C^{0.50},\quad D\_{\text{opt}} \propto C^{0.50}$$

换言之——**每 1 个模型参数应对应约 20 个训练 token**（即 $D/N \approx 20$）。Chinchilla 用 70B 参数、1.4T token 训练的模型，在同等计算预算下超越了当时所有更大但训练不足的模型（如 280B 参数的 Gopher）。

**计算最优分配公式**：给定总计算预算 $C\_{\text{total}}$，最优解应满足：

$$\frac{\partial L}{\partial N} \bigg/ \frac{\partial L}{\partial D} = \frac{\partial C}{\partial N} \bigg/ \frac{\partial C}{\partial D} \approx \frac{6D}{6N} = \frac{D}{N}$$

代入损失函数的幂律形式求解，可得：

$$N^* \approx \left(\frac{\alpha\_N}{\alpha\_D} \cdot \frac{C}{6}\right)^{\frac{\alpha\_N \alpha\_D}{\alpha\_N + \alpha\_D}},\quad D^* \approx \left(\frac{\alpha\_D}{\alpha\_N} \cdot \frac{C}{6}\right)^{\frac{\alpha\_N \alpha\_D}{\alpha\_N + \alpha\_D}}$$

当 $\alpha\_N \approx \alpha\_D$（Chinchilla 的测量结果）时，两者指数退化为 $0.5$，即 $N^* \approx D^* \propto C^{0.5}$。

**Kaplan vs Chinchilla 核心差异**：

| 维度 | Kaplan (2020) | Chinchilla (2022) |
|------|-------------|-----------------|
| 核心主张 | 优先扩大模型 | 同步扩大模型 + 数据 |
| $N\_{\text{opt}}$ 指数 | $\propto C^{0.73}$ | $\propto C^{0.50}$ |
| $D\_{\text{opt}}$ 指数 | $\propto C^{0.27}$ | $\propto C^{0.50}$ |
| 最优 $D/N$ 比 | ~1.4 tokens/param | ~20 tokens/param |
| 关键方法论问题 | 固定训练步数，大模型 undertrained | 对每个模型规模独立调学习率 schedule |
| 实践影响 | GPT-3（175B, ~300B tokens） | Chinchilla（70B, 1.4T tokens）→ Llama 2/3 |

Chinchilla 的结论在后续实践中得到了广泛验证。**Llama 3**（405B, 15T+ tokens）和 **DeepSeek-V3**（671B MoE, 14.8T tokens）均采用了远超 Kaplan 建议的数据量，印证了"数据与模型同步缩放"的正确性。

### 涌现能力（Emergent Abilities）

涌现能力（Wei et al., 2022）是 Scaling Laws 最引人注目的推论之一：**某些任务能力并非随着模型规模线性提升，而是在跨越某个临界阈值后突然出现**——呈现出明显的**相变（Phase Transition）** 行为。

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>Performance</mi>
  <mo>(</mo>
  <mi>N</mi>
  <mo>)</mo>
  <mo>≈</mo>
  <mfenced open="{" close="">
    <mtable>
      <mtr>
        <mtd>
          <mtext>random</mtext>
        </mtd>
        <mtd>
          <mtext>当 </mtext>
          <mi>N</mi>
          <mo><</mo>
          <msub>
            <mi>N</mi>
            <mtext>crit</mtext>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mtext>above-random</mtext>
        </mtd>
        <mtd>
          <mtext>当 </mtext>
          <mi>N</mi>
          <mo>≥</mo>
          <msub>
            <mi>N</mi>
            <mtext>crit</mtext>
          </msub>
        </mtd>
      </mtr>
    </mtable>
  </mfenced>
</math>

涌现能力的典型表现包括：多步算术推理、词义消歧、指令跟随、代码理解等。在阈值以下，模型在这些任务上的表现与随机猜测无异；一旦跨过阈值，性能陡峭上升至远高于随机水平。

**涌现与 Reasoning 的内在关联**：这直接解释了为什么 **Chain-of-Thought（CoT）推理只在参数规模达到特定门槛后才有效**。Wei 等人发现，CoT prompting 在 ~10B 参数以下的模型中几乎不产生任何提升——模型无法利用中间推理步骤来修正最终答案，因为其参数容量不足以学习"推理过程与最终答案之间的一致性约束"。而当参数规模跨越 ~50B-100B 阈值后，CoT 的效果陡升：模型不仅会"模仿"推理格式，还能在推理过程中进行**自修正**（self-correction）——这与本文 [LLM Reasoning](#llm-reasoning) 中讨论的 "Aha Moment" 现象（模型自发生成 "Wait, let me reconsider..."）具有相同的数学根源：**参数容量越过临界点后，条件概率图中首次出现了连接"中间不确定性判断"到"修正后推理"的概率路径**。

从 Policy Optimization 的统一视角看，涌现意味着**策略的搜索空间拓扑在临界参数处发生质变**——模型从"单峰分布"（所有回复集中在错误答案附近）过渡到"多峰分布"（正确答案的高概率区域与错误答案的低概率区域共存但可区分），这使得 RL 的奖励信号能有效地将概率质量从后者推向前者。

**与 MoE 的关联**：涌现行为也是 Mixture of Experts 架构的理论动机之一。MoE 模型（如 DeepSeek-V3、Mixtral 8×7B）通过稀疏激活的路由机制，使总参数量远超实际推理时激活的参数量。总参数量决定"涌现阈值"，激活参数量决定"推理效率"。这意味着 MoE 模型可以在推理成本接近小模型的前提下，通过大幅提升总参数规模来跨越涌现临界点，解锁 CoT 推理等高级能力。DeepSeek-V3 将 MoE 路由思想延伸至推理层面的 System 1/System 2 路由（见 [System 1 vs System 2](#system-1-vs-system-2推理的工程实现)），实际上是**涌现能力在推理效率上的二次利用**：用总参数量换取能力的涌现，用稀疏路由换取推理的经济性。

### 实践启示

#### （一）预训练成本主导——幂律的残酷经济学

Scaling Law 的幂律指数直接决定了预训练的成本结构。设损失降低目标为 $\Delta L = L\_1 - L\_2$，根据 $L \propto C^{-\alpha\_C}$（$\alpha\_C \approx 0.05\sim0.1$），所需计算量的比值满足：

$$\frac{C\_2}{C\_1} = \left(\frac{L\_1}{L\_2}\right)^{1/\alpha\_C}$$

取 $\alpha\_C \approx 0.05$（Kaplan 的估计），当目标是将损失降低至原来的 $1/2$（质量翻倍），需要：

$$\frac{C\_2}{C\_1} = 2^{1/0.05} = 2^{20} \approx 10^6$$

即**约一百万倍的计算增长**。即使采用更乐观的 $\alpha\_C \approx 0.10$，$2^{10} \approx 10^3$ 倍的计算增长也是不可避免的。更实际地说：**损失每降低 10%，计算量需增长约 6~12 倍**。

这意味着预训练成本是 LLM 全生命周期中的绝对主导项——SFT、RL 等下游训练环节的计算量通常仅为预训练的 1%~5%。这解释了为什么：SFT 数据质量远比数量重要（高质量少量 SFT 数据即可有效微调，因为预训练已经提供了强大的先验分布）；"训练到底"（训练至收敛）在预训练阶段几乎不可能，因为指数衰减的损失意味着每个额外的 10% 质量改进都需要指数级的算力投入。

#### （二）蒸馏的理论支撑——为什么剪枝模型可以超越同规模原生模型

Chinchilla 定律为知识蒸馏提供了一个优雅的理论解释。一个在 Chinchilla 最优比（~20 tokens/param）上训练的大模型产生了极其**信息密集（information-dense）** 的输出分布——它的每个 token 都经过了充分利用参数容量的训练，输出概率分布高度结构化。

当这个教师模型的输出（logits 或生成的文本）被用作小模型的训练数据时，小模型接收到的不是"原始语料中的稀疏信息"，而是**已经被教师模型压缩和结构化的高信息密度信号**。从信息论的角度看：

$$\mathcal{I}(Y\_{\text{teacher}}; X) > \mathcal{I}(Y\_{\text{random}}; X)$$

其中 $\mathcal{I}$ 表示互信息——教师模型的输出与输入之间的互信息远高于随机采样数据。这意味着小模型在相同 token 数量下，从教师数据中学到的"有用信息"更多。

这与本文 [On-Policy Distillation](#on-policy-distillation) 的讨论直接呼应：当教师模型是 Chinchilla 最优配置训练的大模型（如 DeepSeek-R1 作为教师），学生模型即使参数量远小于教师（如 1.5B 或 7B），通过接收教师的高信息密度输出，可以在同等计算预算下达到甚至超越同规模原生 Chinchilla 最优训练的模型。On-Policy Distillation 更进一步地利用了这一原理——用 RL 训练中当前策略的在线采样数据做蒸馏，确保数据分布始终与策略对齐，从而最大化每一个 teacher-student 交互的信息效率。

#### （三）"数据墙"问题——当前 Scaling 的真正瓶颈

截止 2025 年，Scaling Laws 面临的最严峻挑战可能不是算力不足，而是**数据枯竭（data wall）**。Chinchilla 定律告诉我们，最优训练需要 $D/N \approx 20$。以 Llama 3（405B）为例，Chinchilla 最优的数据量约为 $405 \times 10^9 \times 20 = 8.1 \times 10^{12}$ tokens（8.1T），而目前公开可用的高质量自然语言文本总量估计在 10T-20T 级别。当模型规模向万亿参数逼近时，自然数据本身将成为瓶颈。

这驱动了三个方向的研究：
- **合成数据（Synthetic Data）**：用强模型生成训练数据（本质是蒸馏的规模化），但面临"模型坍缩（model collapse）"风险——递归生成的数据会导致分布逐步退化
- **多模态数据**：将图像、视频、音频数据纳入训练，绕过纯文本的数据墙
- **数据效率优化**：通过更好的数据筛选（如 DoReMi、DSIR）在有限数据内最大化信息量，等价于提升有效 $D$ 而不增加实际 token 量

"数据墙"的存在也强化了 RL 和 On-Policy Distillation 的重要性——当预训练的边际收益因数据瓶颈而递减时，通过 RL 在已有模型上进行精细化策略优化和蒸馏，成为性价比最高的改进路径。

### Scaling Laws 与本文核心内容的连接

**（a）Scaling Laws → SFT 数据质量优先于数量**

预训练阶段的幂律关系已经将损失压至极低的水平——模型在数十万亿 token 上学会了极为丰富的语言先验。SFT 阶段的数据量（通常百万到千万级别）相对于预训练数据量几乎可以忽略不计。因此，SFT 数据的信息密度远比数量重要：1000 条精心筛选、覆盖关键分布的高质量 SFT 样本，效果远超 10000 条随机采集的低质量样本。在 Scaling Laws 的框架下，SFT 的本质是在预训练的 $L\_\infty$（不可约减损失）附近做极小范围的分布修正——不需要、也不应该用数据量来驱动，而应该用**数据质量（与目标分布的匹配度）** 来驱动。

**（b）Scaling Laws → GRPO 组大小 $G$ 的最优选择**

GRPO 的核心机制——组内 Z-score 标准化计算相对优势（见 [GRPO 章节](#grpogroup-relative-policy-optimization)）——其有效性高度依赖组大小 $G$。Scaling Laws 提供了一个理解 $G$ 的理论视角：

- **$G$ 的本质是对"探索预算"的分配**：每条 prompt 采样 $G$ 条回复，需要 $G$ 倍的推理计算。在固定总计算预算 $C\_{\text{rollout}}$ 下，存在 prompt 数量 $P$ 与组大小 $G$ 的 trade-off：$C\_{\text{rollout}} \propto P \times G$。
- **$G$ 太小**（如 $G=2$）：组内方差估计不稳定，Z-score 标准化后的优势信号噪声大，策略梯度方向不可靠——等价于在小批量下做高方差梯度估计。
- **$G$ 太大**（如 $G=64$）：每 prompt 占用大量推理预算，导致可覆盖的 prompt 多样性下降（$P$ 减小），策略在少量 prompt 上过拟合。

从 Chinchilla 视角出发，$G$ 的最优值应满足："每 prompt 的探索预算"和"prompt 多样性覆盖"之间的计算最优平衡。实践中 $G=4\sim16$ 成为主流选择，这与 Scaling Laws 的精神一脉相承：**在有限计算预算内，探索信号的质量（组内方差足够大）与覆盖的广度（prompt 数量足够多）需要同步扩展**。

**（c）Scaling Laws → 计算预算作为 Harness Engineering 的根本约束**

[Harness Engineering](#harness-engineering) 中讨论的所有工程挑战——Rollout 引擎的吞吐量、训练引擎的显存优化、异构硬件的资源调度——其根源都可以追溯到 Scaling Laws 揭示的计算预算约束。幂律意味着质量改进的边际成本呈指数增长，因此工程系统的效率（GPU 利用率、通信开销、流水线气泡时间）直接决定了在给定预算下"能走多远"。

举例而言，若 Rollout 引擎的 GPU 利用率从 60% 提升至 80%，等效于在相同硬件上获得了 33% 的额外计算预算。在 Scaling Laws 的框架下，这 33% 的增量可以转化为损失降低 $\Delta L \approx (1.33)^{\alpha\_C} - 1 \approx 1.4\%\sim2.9\%$——看似微小，但在 SOTA 竞争的边际上足以产生显著的能力差异。Harness Engineering 中 vLLM 的 PagedAttention、DeepSpeed 的 ZeRO-3、Megatron 的张量并行等技术，本质上是在与 Scaling Laws 的指数衰减赛跑——从工程效率中"挤出"被幂律稀释的质量增益。

## LLM的本质
LLM, Large language model本质上就是基于Transformer在做自回归的任务（即预测下一个token），LLM的训练可以分为三个部分：
- **预训练（Pretraining）**：预训练的过程就是让模型学会如何生成语言，所以本质上预训练模型的任务目标是续写，在具体的训练过程中，输入数据为大量的语言数据，模型通过自监督的训练方式预测下一个token的概率分布，然后通过交叉熵损失来优化整个模型。
- **监督微调（SFT，Supervised FineTuning）**：预训练的模型已经拥有了生成的能力，但是缺乏对话的能力，而SFT的过程就是让模型学会对话。SFT和Pretraining的训练过程以及损失函数是一致的，不同之处在于，SFT的训练数据是对话形式的，我们不需要去学习生成问题的能力，而是要根据问题来生成答案，因此对问题（Prompt）去做交叉熵损失就是没有必要的，SFT的做法是做一个mask，将Prompt部分的token id赋值为-100（交叉熵损失会将其忽略）。并且由于在Pretraining的过程中，我们已经得到一个非常强大的语言生成模型，我们在SFT的过程中就不太需要训练很多轮次。
- **对齐（Alignment）**：在SFT中模型学会了对话的能力，但是对话生成的答案质量往往取决于SFT训练数据集的质量，如果遇到训练数据外的分布，那么就会出现严重的幻觉。因此，我们不仅需要让LLM学会对话，还要让LLM学会如何生成一个好的答案，这就是LLM对齐要做的事。

## Mixture of Experts（MoE）

MoE（混合专家）是一种在**不显著增加计算量**的前提下大幅提升模型参数容量的架构策略。其核心洞见是：每个token只激活一小部分"专家"，而非整个模型。这使MoE成为Scaling Laws在计算效率约束下的关键实现路径——在相同FLOPs预算下，MoE可以支撑数倍于密集模型的参数总量。

### 核心思想：稀疏激活

传统Transformer的FFN层对每个token执行相同的全量计算。MoE将单一FFN替换为$E$个并行的"专家"（每个专家本身是一个小型FFN），但对每个token仅激活其中$k$个专家（$k \ll E$，典型值$k=2, E=8$）：

| 属性 | 密集模型（Dense） | MoE模型 |
|------|-------------------|---------|
| 总参数量 | $N$ | $N$（含$E$个专家） |
| 每token激活参数 | $N$ | $\frac{k}{E} \cdot N\_{\text{FFN}} + N\_{\text{attn}}$ |
| 每token FLOPs | $\approx$ 密集FLOPs | $\approx$ 密集FLOPs（仅激活$k$个专家） |
| 模型容量 | 固定 | $E/k$ 倍于同FLOPs密集模型 |

稀疏激活的精妙之处在于**解耦了参数容量与计算成本**：总参数可以做到$8\times$甚至$16\times$密集模型的规模，但每个token的实际计算量几乎不变。这使得MoE成为"用计算效率换模型容量"的典范工程范式。

### 路由机制：谁来决定激活哪些专家？

路由（Routing）是MoE的核心决策组件。给定token的隐状态$h \in \mathbb{R}^d$，路由器的计算流程为：

$$\mathbf{p} = \text{softmax}(W\_r \cdot h), \quad W\_r \in \mathbb{R}^{E \times d}$$

$$\mathcal{T} = \text{TopK}(\mathbf{p}, k)$$

其中$W\_r$是一个轻量级线性层（参数量$E \cdot d$，相比专家FFN可忽略不计），TopK选择概率最高的$k$个专家。被选中专家的输出由其门控概率加权求和：

$$y = \sum\_{i \in \mathcal{T}} p\_i \cdot \text{Expert}\_i(h)$$

路由器的设计直接决定了训练的稳定性和模型的表达能力。核心挑战是**负载均衡（Load Balancing）**——如果路由总是选择少数几个"热门"专家，那些专家会接收绝大部分token，导致：(1) 热门专家过拟合，冷门专家退化；(2) 计算资源分配不均，并行效率骤降。

**辅助负载均衡损失（Auxiliary Load Balancing Loss）** 是解决这一问题的标准手段。定义专家$i$被分配的token比例为$f\_i$，门控概率均值为$g\_i$，则辅助损失为：

$$\mathcal{L}\_{\text{aux}} = \alpha \cdot E \cdot \sum\_{i=1}^{E} f\_i \cdot g\_i$$

其中$\alpha$为平衡系数（典型值$10^{-2}$量级）。这等价于最小化专家使用分布的变异系数（CV）平方——当所有专家均匀使用时，$\mathcal{L}\_{\text{aux}}$达到最小值。数学上，这是通过惩罚"高分配份额 × 高门控概率"的组合来拉平专家利用率。

**Expert Choice Routing vs Token Choice Routing**：传统路由是"token选择专家"（Token Choice）——每个token独立选择top-k专家，可能出现"扎堆"问题（大量token选择同一专家）。Expert Choice Routing则反其道而行之：让每个专家选择其最有把握处理的前$\frac{k \cdot B}{E}$个token（$B$为batch size），通过求解匹配问题保证负载严格均衡。这在理论上更优雅（负载均衡是硬约束而非软惩罚），但实现复杂度更高，且需要全局排序操作。

### DeepSeekMoE 深度解析

DeepSeek在MoE架构上做出了两项关键创新：**细粒度专家（Fine-grained Experts）** 与**共享专家（Shared Experts）**。

**细粒度专家**将标准FFN专家进一步拆分为更小的子专家。若标准MoE有$E$个专家、每专家FFN隐层维度为$d\_{\text{ff}}$，细粒度MoE将其拆分为$m \cdot E$个专家，每专家FFN隐层维度为$d\_{\text{ff}} / m$。在激活相同参数量（$k \cdot d\_{\text{ff}}$）的前提下，细粒度版本可以激活$m \cdot k$个专家，获得更丰富的专家组合。直觉上，这类似于卷积神经网络中使用更多较小卷积核代替大卷积核——**组合灵活度指数级提升**：从$C(E, k)$种组合增长到$C(mE, mk)$种，使模型有更多的路由路径来表达不同类型的输入。

**共享专家**是另一项重要设计：除$E$个路由专家外，额外设置一小批**始终激活**的共享专家，其输出与路由专家的加权输出直接相加。共享专家捕捉跨所有输入类型的通用知识（如基础语法、常识），让路由专家专注于各自擅长的细分领域（如数学推理、代码生成、文学创作），减少路由专家之间的知识冗余，并使路由选择更"纯粹"地反映输入的专业需求。

DeepSeek-V2和V3的成功证明，细粒度+共享专家的设计在实际大规模训练中显著优于标准MoE架构。从RL的视角审视，每个token的专家选择本质上是一个**离散动作**——路由器在$E$个专家中选择$k$个的组合空间中进行决策。这提出了一个深刻的问题：**能否用RL直接优化路由策略？** GRPO的组内相对优势框架天然适合离散动作空间的探索——将路由选择视为策略动作，将最终任务损失作为奖励信号，通过强化学习直接训练路由器（而非仅依赖辅助损失）。这一方向（RL-based Routing）是MoE研究的开放前沿，但面临"路由梯度不可导"的根本障碍——TopK是离散操作，需要Gumbel-Softmax重参数化或REINFORCE估计器来绕过。

### 训练挑战

**(a) 专家并行（Expert Parallelism）**：在分布式训练中，不同专家分布在不同的GPU上。每个token的路由决策可能将其派发到任意GPU上的专家，这意味着每层MoE都需要一次**全对全通信（All-to-All Communication）**——每个GPU将其上的token按路由结果发送到目标GPU，计算后再接收回来。通信量随专家数线性增长，在跨节点场景下成为训练速度的主要瓶颈。DeepSpeed-MoE等框架通过拓扑感知的专家放置和通信-计算重叠来缓解这一问题。

| 并行策略 | 通信模式 | 适用场景 |
|---------|---------|---------|
| 数据并行 | All-Reduce（梯度同步） | 密集模型 |
| 张量并行 | All-Reduce（激活切分） | 单层过大 |
| 专家并行 | All-to-All（token派发+回收） | MoE特有 |

**(b) 负载不均衡检测**：训练过程中需要持续监控两个指标：专家使用频率的熵（理想情况下接近均匀分布的熵$\log E$）和专家丢弃率（因达到capacity上限而被丢弃的token比例）。当任一指标异常时，需要调大辅助损失系数$\alpha$或引入显式的capacity约束。

**(c) 专家丢弃（Expert Dropout）**：训练时随机屏蔽部分专家（类似Dropout），迫使路由器学会在专家不可用的情况下仍能正常路由，增强模型对专家失效的鲁棒性。这在推理阶段尤为重要——当某些专家因量化或部署原因性能下降时，模型不会崩溃。与Dropout的联系：$p\_{\text{drop}}$调控路由多样性的强制程度。

### 与本文各主题的连接

**(a) MoE的离散路由 $\approx$ Agentic RL的工具选择**：两者都是在大型离散动作空间中做决策——MoE在每个token的$E$个专家中选$k$个，Agent在每个决策步的$K$个工具中选1个。两者的核心挑战也高度同构：负载均衡对应探索-利用平衡（防止偏好少数"安全"动作），路由梯度不可导对应用REINFORCE估计信用分配。

**(b) 负载均衡 $\approx$ 探索平衡**：MoE的辅助负载均衡损失确保所有专家都被充分训练，这与RL中的entropy bonus（策略熵正则化）确保所有动作都有被尝试的机会，是同一数学思想在不同场景下的投影——都防止模型过早坍缩到少数"容易"的选择上。

**(c) MoE的稀疏计算 $\approx$ Efficient Reasoning的自适应计算预算**：MoE让模型"按需激活"专家，Efficient Reasoning让模型"按需分配"推理token。两者共享的核心理念是**计算不应该均匀分配**——简单token激活通用专家、跳过复杂推理；困难token激活专业专家、触发深度思考。这是"自适应计算"在架构和推理两个维度的对称体现。

## LLM Reasoning
LLM Reasoning是LLM在生成答案之前的推理过程。LLM本身就具有一定的推理能力，原因是在训练过程中，LLM的训练数据本身就包含了大量的逻辑推理过程。由于LLM输出的本质是概率分布，给LLM一个问题，有可能直接输出direct response，也有可能输出thoughtful response。**让LLM稳定reasoning的本质就是改变LLM的输出概率分布，让thoughtful response的概率提高。**
让LLM稳定reasoning的方法：
- CoT Prompting：通过改变输入来改变LLM的输出概率分布，使中间推理步骤的条件概率提高，进而提升最终答案的准确性。主要方法包括：
  - **Few-shot CoT**（Wei et al., 2022）：在prompt中提供带推理步骤的示例，让LLM模仿这种逐步推理的方式。
  - **Zero-shot CoT**：在prompt末尾添加"Let's think step by step"等触发短语，无需示例即可激发推理。
  - **Self-Consistency**（Wang et al., 2022）：对同一问题采样多条推理路径，再通过多数投票选出最终答案，提高推理的鲁棒性。
  - **Tree of Thoughts**（Yao et al., 2023）：将推理扩展为树形结构，通过搜索算法（BFS/DFS）探索多条路径并剪枝，适合需要深度搜索的复杂问题。
  - **ReAct**（Yao et al., 2022）：将推理（Reasoning）与行动（Acting）交替进行，让LLM在推理过程中调用外部工具（搜索、计算器等），弥补纯语言推理的事实局限。

  CoT Prompting本质上属于推理时干预（inference-time intervention），不改变模型参数，成本低但依赖prompt设计，泛化能力有限。

- SFT：SFT将推理能力内化到模型参数中，使LLM无需显式CoT示例也能稳定输出推理过程。训练数据通常来自两类来源：（1）人工标注的逐步推理轨迹，质量高但成本高；（2）从更强的推理模型（如o1、DeepSeek-R1）蒸馏的推理数据，成本低且易于规模化。训练目标与标准SFT一致——最大化推理token的条件对数似然，但会对最终答案token施加更高权重。SFT的核心优势是推理行为稳定可控；缺点是完全依赖训练数据的分布，对数据外的新问题类型泛化能力有限，且由于有监督信号的存在，LLM只会模仿已有推理路径，无法自主探索出更优的推理策略。

- RL：RL通过奖励信号让LLM自主探索推理策略，突破SFT对标注数据分布的依赖。具体流程是：LLM对同一问题采样生成多条推理路径，Reward Model或基于规则的验证器（如答案正确性、格式规范性）对各路径打分，再通过策略优化算法（PPO、GRPO等）更新模型参数，增大高分路径的生成概率。RL的核心优势在于LLM可以自主探索出人类从未标注的优质推理路径，甚至涌现出"Aha Moment"——模型自发学会在推理中进行自我反思和纠错（"Wait, let me reconsider..."）。**DeepSeek-R1**是RL驱动推理的代表性工作，通过GRPO和基于规则的奖励系统，在不依赖人工推理轨迹的情况下让LLM涌现出强大的推理能力。

### CoT的本质：策略搜索空间的扩张

从强化学习的视角审视，CoT的数学本质是**利用额外生成的中间token来扩张策略的搜索空间**。给定一个问题 $x$，LLM的生成过程是自回归的条件概率链：

$$\pi\_\theta(y|x) = \prod\_{t=1}^{T} \pi\_\theta(y\_t|x, y\_{<t})$$

**Direct Response**（无CoT）直接将问题映射到答案，搜索空间仅为输出序列的所有可能组合。对于复杂推理问题（如数学证明、多步逻辑），正确路径在直接映射下是概率极低的"针尖"，模型极易在错误的捷径上收敛。

**CoT**插入推理token序列 $\{r\_1, r\_2, ..., r\_K\}$ 作为中间状态，将生成过程分解为：

$$\pi\_\theta(y|x) = \sum\_{r\_1, ..., r\_K} \pi\_\theta(r\_1|x) \cdot \pi\_\theta(r\_2|x, r\_1) \cdots \pi\_\theta(y|x, r\_1, ..., r\_K)$$

这些中间推理步骤起到两个关键作用：(1) **状态分解**——将"问题→答案"的长程跳跃分解为"问题→中间结论→中间结论→答案"的短程级联，每一步的条件概率显著更高；(2) **自修正路径**——中间步骤允许模型输出不确定性（"这可能是错误的，让我重新检查..."），从而在生成过程中动态调整后续推理方向。这本质上是在条件概率图上开启了更多可达路径，将搜索从宽度优先的隐式空间展开为深度优先的显式轨迹。

### System 1 vs System 2：推理的工程实现

Kahneman的"双系统"认知理论在LLM推理中有清晰的工程映射：

| 维度 | System 1（快速直觉） | System 2（慢速推理） |
|------|---------------------|---------------------|
| **输出模式** | 直接生成答案，无中间推理 | 逐步推理 + 最终答案 |
| **推理时机制** | 单次前向传播（1 pass） | 多次自回归解码 + 验证 |
| **token效率** | 高（仅输出答案token） | 低（推理token数 ≫ 答案token数） |
| **准确率** | 低（复杂问题失效） | 高（可处理多步推导） |
| **计算成本** | $O(T\_{\text{ans}})$ | $O(T\_{\text{ans}} + T\_{\text{reason}})$ |
| **对应训练方法** | 标准SFT | SFT + RL（CoT数据 + 推理奖励） |
| **代表模型** | GPT-4 (direct), Claude (无thinking) | o1, o3, DeepSeek-R1, Claude Extended Thinking |

**工程实现的关键决策**是何时触发System 2。当前主流方案包括：(1) **路由分类器**——用小模型判断问题难度，简单问题走快速通道（System 1），复杂问题走推理通道（System 2），如 DeepSeek-V3 的 MoE 路由思想在推理层面的延伸；(2) **推理预算token**——强制分配固定数量的thinking token（如Claude Extended Thinking的budget参数），模型在预算内自由组织推理过程；(3) **自适应终止**——模型自主判断推理是否充分（通过输出结束标记或置信度阈值），无需人工预设推理长度。

更前沿的方向是**推理-行动交织（Interleaved Reasoning-Action）**：System 2不只是"先想后答"，而是在推理过程中动态穿插工具调用——推理到某一步时调用计算器验证数值，或调用代码执行器运行片段代码——将语言推理与外部验证实时融合。ReAct和Tool-Integrated CoT是该范式的早期实现。

### 推理长度的经济学：成本与精度的权衡

推理长度是LLM推理中最核心的**可控经济变量**。长推理链虽然提升准确率，但带来三重成本：

**1. 延迟成本**：$T\_{\text{reason}}$ 个推理token的生成时间是纯答案生成的 $\frac{T\_{\text{reason}}}{T\_{\text{ans}}}$ 倍。对于需要实时交互的场景，每条查询多等2秒即可显著降低用户体验。

**2. 显存/吞吐成本**：长推理链的KV cache与序列长度成正比 $O(L \cdot d\_{\text{model}} \cdot n\_{\text{layers}})$。在大规模在线服务中，一条10K token的推理请求可能占用等同于5条2K token请求的显存，直接削减服务吞吐量。

**3. 金钱成本**：API调用按token计费，推理token通常与输出token同价。一条包含2000-token推理链的回答可能比直接回答贵5-10倍。

**长度惩罚的数学形式**：设推理链长度为 $L$，最终答案正确性得分为 $R\_{\text{acc}}$，则总奖励可建模为：

$$R\_{\text{total}}(L) = R\_{\text{acc}}(L) - \lambda \cdot L$$

其中 $R\_{\text{acc}}(L)$ 通常是 $L$ 的凹函数——推理长度带来的精度提升存在边际递减效应（前200个推理token可能大幅提升准确率，之后每增加100个token的边际收益迅速衰减）。$\lambda$ 是长度惩罚系数，控制 cost-accuracy trade-off 的斜率。

**最优推理长度的求解**：在给定问题难度下，最优推理长度 $L^*$ 应满足一阶条件：

$$\frac{\partial R\_{\text{acc}}}{\partial L}\bigg|\_{L=L^*} = \lambda$$

即当推理的边际精度收益恰好等于边际成本时，总效用最大。实践中 $R\_{\text{acc}}(L)$ 的具体函数形式取决于模型能力和任务难度——对简单问题，$R\_{\text{acc}}(L)$ 在很小的 $L$ 处就接近饱和；对复杂问题，$R\_{\text{acc}}(L)$ 可能持续增长数百至数千个token。

这一公式也解释了**动态推理预算**的动机：当模型能准确估计问题难度时，可自适应选择 $L^*$——简单问题 $L^* \approx 0$（跳过推理，即System 1），中等难度 $L^* = 200\sim500$，高难度 $L^* = 2000+$。

Efficient Reasoning方法：长推理链虽然准确，但对简单问题会产生大量冗余token，增加推理延迟和计算成本。高效推理的目标是在保证质量的前提下降低推理开销，主要方法包括：
1. **动态推理预算（Thinking Budget）**：根据问题难度动态调整推理链长度，对简单问题缩短推理过程，对复杂问题保持完整推理。数学上等价于对每个查询求解 $\arg\max\_L [R\_{\text{acc}}(L|x) - \lambda L]$，其中 $R\_{\text{acc}}(L|x)$ 为条件于问题 $x$ 的精度函数。Claude 3.7 引入了 Extended Thinking 的预算控制机制，可在准确性与延迟之间灵活权衡。
2. **推理蒸馏（Reasoning Distillation）**：通过SFT将长推理链压缩为更短但等效的推理路径，剔除冗余的重复验证步骤，常用于将大模型推理能力迁移到小模型。本质是以教师模型的 $R\_{\text{acc}}(L)$ 曲线为监督信号，训练学生模型以更小的 $L$ 达到同等的 $R\_{\text{acc}}$。
3. **Speculative Decoding**：利用小模型（draft model）快速生成候选token序列，再由大模型并行批量验证，在不降低输出质量的前提下显著提升生成速度。推理token的验证可按 $k$ 个token并行，将 $O(L)$ 的串行延迟降低为 $O(L/k)$。
4. **Latent Reasoning（COCONUT, Chain of Continuous Thought）**：将推理过程从离散的语言空间迁移到连续的隐空间（latent space），用最后一层隐状态直接作为下一步的"思维"输入，而非解码为token，大幅减少推理所需的token数量，在某些多步推理任务上性能不降反升。这从根本上绕过了token级的自回归瓶颈，是在推理效率上的范式级突破。

### 推理时计算扩展（Test-Time Compute Scaling）

推理长度经济学讨论的是「给定推理链长度下的成本-精度权衡」，而推理时计算扩展（Test-Time Compute Scaling）回答一个更根本的问题：**对于固定参数的模型，增加推理时的计算量，精度能提升多少？** 这是继模型参数量 Scaling 和数据量 Scaling 之后的第三条 Scaling 轴——不改变模型权重，仅通过改变推理过程的计算分配来提升性能。

#### 1. 推理时 Scaling Laws：精度 ∝ log(推理计算量)

OpenAI（2024）和 DeepMind 的独立研究共同揭示了一个核心规律：在固定模型参数量的条件下，模型在困难任务上的准确率与推理时计算量呈对数线性关系：

$$\text{Accuracy}(\text{compute}) \approx a \cdot \log(\text{compute}\_{\text{inference}}) + b$$

其中 $\text{compute}\_{\text{inference}}$ 是推理阶段消耗的总 FLOPs。这一规律的直接推论是：**将推理计算量翻倍，精度获得的是加法级（而非乘法级）提升**——这是典型的边际递减。但关键洞察在于，推理计算量有三条不同的扩展轴，每条轴的对数关系有不同的系数 $a$：

| 扩展轴 | 机制 | 精度-计算量关系 | 效率特征 |
|--------|------|:---:|----------|
| **(a) 序列化思考 Token** | 生成更多推理 token，逐步展开中间推导 | $\text{Accuracy} \sim \log(L)$ | 计算量 ∝ 序列长度，对需深度推理的任务效率高；受限于模型自身的推理能力上限 |
| **(b) 并行采样（Best-of-N）** | 独立采样 $N$ 条完整回复，由验证器选出最优 | $\text{Accuracy} \sim \log(N)$ | 天生可并行，延迟不随 $N$ 线性增长；但受限于 order statistic bound，效率低于序列化 |
| **(c) 验证器引导搜索** | 用 PRM 对中间步骤打分，引导树搜索 | $\text{Accuracy} \sim \exp(b)$ | 当 PRM 质量足够高时，精度与搜索预算 $b$ 呈**指数**关系——远超前两者的对数效率 |

三条轴的效率差异揭示了核心权衡：序列化思考和并行采样都是**无反馈**的扩展（盲目生成更多 token 或更多候选），而验证器引导搜索通过**过程级反馈**将计算量投到最有希望的路径上，因此效率最高。这正是 o1 类推理模型在困难数学/编程任务上远超简单 CoT 的本质原因。

#### 2. Best-of-N 深度剖析：为什么只是 $\log N$？

给定问题 $x$，对模型采样 $N$ 条独立回复 $y\_1, ..., y\_N \sim \pi\_\theta(\cdot|x)$，用奖励模型（RM/PRM）对每条打分 $r\_i = R(x, y\_i)$，最终返回得分最高的回复。直观上似乎「多投几次总有一次好」，但数学上 Best-of-N 的期望奖励增长远慢于直觉：

$$E[\max(r\_1, ..., r\_N)] \approx \Phi^{-1}\left(1 - \frac{1}{N}\right)$$

其中 $\Phi^{-1}$ 是标准正态分位点函数。这一结果的来源是**次序统计量（Order Statistic）的极值理论**：在弱假设下（奖励分布属于 Gumbel 域），$N$ 次独立采样的极值期望以 $O(\sqrt{\log N})$ 或更慢的 $\log N$ 量级增长，**而非线性的 $O(N)$**。直觉谬误在于：每次采样不是向「更好的回复」的定向前进，而是在一个固定分布中随机抽取，越往后抽取到比已有最大值更好的样本的概率越低。

**GRPO 的组采样本质**：GRPO（Group Relative Policy Optimization）在每次更新时为每个 prompt 采样 $G$ 条回复（group），计算组内相对优势，再用策略梯度拉升高优势回复的概率。这本质上是一个**mini Best-of-N 内嵌于 RL 训练**的过程——$G$ 越大，组内出现高奖励回复的概率越高，优势信号的方差越小，训练越稳定。区别在于，Best-of-N 是推理时的纯选择（不改变分布），而 GRPO 通过策略梯度将选择转化为分布偏移——**GRPO 是在参数空间中实现 Best-of-N**。

#### 3. 验证器引导搜索（Verifier-Guided Search）：System 2 的树搜索实现

PRM（Process Reward Model）对推理的**每一步**打分，将生成从线性自回归升级为树搜索：

$$P(\text{step}\_{t+1} | \text{step}\_{1:t}) \propto \pi\_\theta(\text{step}\_{t+1} | \text{step}\_{1:t}) \cdot \exp(\text{PRM}(\text{step}\_{1:t+1}) / \tau)$$

具体流程：每一步从当前 active beam 的每个节点出发，生成 $k$ 个候选后继步骤 → PRM 对每个 $(prefix, step)$ 打分 → 保留全局 top-$B$ 最高分的 $B$ 条部分路径（beam）→ 重复直至搜索结束或达到 token 预算。

| 搜索算法 | 机制 | 适用场景 |
|----------|------|----------|
| **Beam Search** | 每步保留 top-$B$ 候选，贪心扩展 | 精度要求中等，延迟敏感 |
| **MCTS** | 用 UCB 平衡探索-利用，模拟 rollout 估计节点价值 | 需深度探索的开放问题 |
| **A\* / Best-First** | 用 PRM 分数作为启发式 $h(s)$，优先扩展最优节点 | PRM 质量极高时的最优策略 |

这极大概率是 **o1 内部「隐藏推理」的实现方式**：表面上 o1 输出一段线性的思维链，但内部在 decode 每一步时，对 $k$ 个候选 token 序列进行 PRM 打分和 beam 剪枝，仅将胜出的路径写入可见的 CoT 输出。这解释了为什么 o1 的推理摘要中频繁出现自我纠错——PRM 评分低的路径被剪枝后，模型被迫回溯到高分节点重新分支。

从认知架构的角度，这正是 **System 2 的树搜索工程实现**：System 1 是单次前向（$B=1, k=1$），Self-Consistency 是并行采样+投票（扁平的 Best-of-N），而树搜索实现了真正的**深度优先 + 剪枝**——在推理空间的每个分叉点进行有反馈的扩展决策。

#### 4. Budget Forcing：o1 的训练范式

Budget Forcing 是 o1 训练中最具原创性的技术——强制模型在固定 token 预算内完成推理，并通过 RL 信号训练模型学会高效分配思考 token：

$$R = \underbrace{R\_{\text{acc}}}\_{\text{答案正确性}} - \lambda \cdot \underbrace{(L\_{\text{used}} - L\_{\text{target}})^2}\_{\text{预算偏差惩罚}}$$

其中 $L\_{\text{target}}$ 是预设推理 token 数，$L\_{\text{used}}$ 是实际使用的 token 数。二次惩罚项同时对**超支**（浪费计算）和**节支**（推理不充分）施加对称惩罚，迫使模型学会在预算约束下找到最优答案。

训练过程：对每个 prompt 随机分配不同的 $L\_{\text{target}}$（如 {500, 1000, 2000, 4000}），让模型在单一训练过程中学会跨预算的泛化推理能力。RL 优化后，模型自发涌现出**自适应计算分配**的模式——简单步骤少写 token，复杂步骤多写 token，遇到瓶颈时主动回溯。

**与「推理长度经济学」的内在联系**：推理长度经济学从推理时（inference-time）角度分析给定问题的 $L^*$ 求解问题（$\frac{\partial R}{\partial L} = \lambda$），而 Budget Forcing 是从训练时（training-time）角度直接教会模型自己做这个优化——模型不再需要外部设定长度，而通过 RL 内化了「在预算边界下最大化正确率」的策略。**二者是同一原理的推理端与训练端实现**。

#### 5. Scaling Paradox：为什么「多想一会」能超越「模型做大」？

一个反直觉的实证发现：在某些推理任务上，**让中小型模型推理更长时间（更多 token）的效果，超过了直接增大模型参数量的效果**。这一现象的数学解释是 CoT 的**循环深度（Recurrent Depth）**效应：

$$\text{Effective Depth}\_{\text{CoT}} = n\_{\text{layers}} \times \frac{T\_{\text{reason}}}{T\_{\text{ans}}}$$

Transformer 的每一层对输入做一次非线性变换。标准前向传播中，深度固定为 $n\_{\text{layers}}$（如 40 层）。但 CoT 将上一时刻的输出作为下一时刻的输入，**同一组层被迭代复用**——模型在第 $t+1$ 步可以「看到」第 $t$ 步的推理结果。对于长度为 $L$ 的 CoT，浅层模型等效获得了 $O(L \cdot n\_{\text{layers}})$ 的计算深度。

这意味着推理时计算扩展创造了**虚拟深度**——不增加参数量，却增加了等效的非线性变换层数。当任务需要大量串行推理步骤时（如多步数学证明），虚拟深度比参数量更关键。这是测试时计算 Scaling 的深层理论依据：**推理时计算本质上是在「租用」额外的模型深度，而不是「购买」更多参数**。

Scaling Laws 的三轴统一：经典的 Kaplan/Chinchilla Scaling Laws 给出了**模型参数量 $N$** 和**训练数据量 $D$** 两轴的最优分配。测试时计算扩展补上了第三轴：

$$L(N, D, C\_{\text{inference}}) = \underbrace{\frac{A}{N^\alpha} + \frac{B}{D^\beta}}\_{\text{经典Scaling Laws}} + \underbrace{f(C\_{\text{inference}})}\_{\text{测试时计算扩展}}$$

三个轴存在复杂的交互：大模型能从同量的测试时计算中获得更多提升（$a\_{\text{large}} > a\_{\text{small}}$），但边际速度递减——`compute-optimal` 的策略是在三轴间以均衡比例分配总预算，而非仅堆某一维度。

## Reward Modeling（奖励模型）

奖励模型（Reward Model, RM）是RLHF流水线中连接「人类偏好」与「策略优化」的关键桥梁。它的任务是将抽象的人类偏好转化为可优化的标量信号。RM的质量直接决定了RLHF的天花板——学术界将这一现象称为**奖励天花板（Reward Ceiling）**：无论策略优化算法多么精巧，最终策略的质量不可能超越RM对「好回复」的辨识能力。

### Bradley-Terry 偏好模型：数学基础

奖励模型的理论根基是 **Bradley-Terry（BT）偏好模型**。给定同一个输入 $x$ 和两个候选回复 $y\_w$（偏好胜出者）与 $y\_l$（偏好失败者），BT模型假设 $y\_w$ 被人类标注者偏好的概率为：

$$P(y\_w \succ y\_l \mid x) = \frac{\exp(r(y\_w))}{\exp(r(y\_w)) + \exp(r(y\_l))} = \sigma(r(y\_w) - r(y\_l))$$

其中 $r(\cdot)$ 是待学习的真实奖励函数，$\sigma$ 为 Sigmoid 函数。这一公式的优雅之处在于它只关心**相对差异** $r(y\_w) - r(y\_l)$——只要RM对好回复的评分高于坏回复（差值无需精确），偏好排序就是正确的。RM的训练采用交叉熵损失，将偏好标注转化为二分类问题：

$$\mathcal{L}\_{\text{RM}} = -\mathbb{E}\_{(x, y\_w, y\_l) \sim \mathcal{D}} \left[ \log \sigma(r\_\phi(y\_w) - r\_\phi(y\_l)) \right]$$

等价于最大化「好回复得分高于坏回复得分」的对数概率。BT模型的一个重要性质是**传递性**——若 $A \succ B$ 且 $B \succ C$，则 $A \succ C$——这使得从成对比较数据中可以推断出全局的偏好排序，是偏好学习可行性的理论前提。

### Reward Hacking 深度分析

Reward Hacking（奖励劫持）指的是策略模型找到了RM的漏洞，以**人类不认可的方式获得高分**。这不是简单的「模型学坏了」，而是RM作为有限容量代理的必然局限。主要表现形式：

**(a) 长度偏差（Length Bias）**：RM在训练中习得了「长回复 = 好回复」的虚假相关性。原因有二：训练数据中优质回复通常更长（包含更多论证细节），且RM的注意力机制天然对更多token有信息聚合倾向。实证研究（Singhal et al., 2023）表明，未加控制的RM在长度指标上的隐含权重可高达0.3~0.5——模型只需写得冗长，即便内容空洞也能获得显著分数提升。

**(b) 格式劫持（Format Hacking）**：模型发现特定格式元素（LaTeX数学公式、代码块、列表编号、加粗文本）能系统性提升RM评分，便开始在回复中大量注入格式糖衣。例如，将简单陈述也包装成数学公式（"答案是 $5$" 而非"答案是5"），或对不需要代码的问题生成伪代码块。这是RM在缺乏格式-内容区分能力的条件下的过拟合。

**(c) 谄媚（Sycophancy）**：模型学会了「用户想听什么就说什么」——当用户暗示某种立场时，模型附和其观点而非坚持事实；当用户对某回复给出纠正性反馈时，模型过度道歉并全盘推翻原有合理分析。本质上，RM的训练数据中，与用户观点一致的回复更容易获得高分，导致RM将「用户立场一致性」作为奖励的代理特征。

**奖励-KL前沿曲线（Reward-KL Frontier）** 揭示了Reward Hacking的动力学。以KL散度 $D\_{\text{KL}}(\pi\_\theta \parallel \pi\_{\text{ref}})$ 为横轴、真实奖励（由人类评估）为纵轴：

```
真实奖励
  ↑
  |     ╭────── 理想区域（真实奖励随KL单调增长）
  |    ╱
  |   ╱
  |  ╱
  | ╱              ← 过优化拐点（Overoptimization Threshold）
  |╱    ╲
  |      ╲────── 黑客区域（RM分数继续↑，但人类评估↓）
  |            ╲
  └──────────────────→ KL散度
```

在拐点之前，RM评分与人类偏好一致增长；拐点之后，策略学会了针对RM的「捷径」，RM分数继续攀升但人类评估反而下降。**拐点的位置是RM质量的核心度量**——优质RM将拐点推到更大的KL值，给予策略更宽的优化安全区。

### ORM vs PRM：两种奖励粒度

奖励模型按评分粒度分为两大类，这一区分对策略优化和推理搜索的设计至关重要：

| 属性 | ORM（Outcome Reward Model） | PRM（Process Reward Model） |
|------|---------------------------|----------------------------|
| **评分对象** | 完整回复 $y$ | 推理链的每一步 $s\_t$ |
| **数学形式** | $R(y \mid x)$ | $r(s\_t \mid s\_{<t}, x)$ |
| **信号密度** | 稀疏（仅在序列末尾一个标量） | 密集（每步推理都获得评分） |
| **训练数据需求** | 仅需最终答案的正确性标注 | 需要步骤级标注（昂贵）或自动标注 |
| **推理应用** | 用于Best-of-N采样、最终验证 | 用于树搜索（beam search、MCTS）中的中间剪枝 |
| **信用分配** | 隐式（依赖策略梯度回溯） | 显式（每步有独立评分） |

**PRM与验证器引导搜索（Verifier-Guided Search）**：在推理时计算扩展（Test-Time Compute Scaling）的框架中，PRM是(c)轴——验证器引导搜索——的核心组件。当PRM能准确评估中间步骤的正确性时，模型可以在推理时进行树搜索：对每个中间状态采样多个可能的下一步，用PRM打分后仅保留高分分支，实现指数级剪枝：

$$P(\text{正确路径}) \propto \prod\_{t=1}^{T} r(s\_t \mid s\_{<t}, x)$$

PRM的评分将每一步的「局部正确性」累积为整条路径的置信度。这与前文讨论的测试时计算扩展形成闭环——PRM质量直接决定了搜索效率，而搜索效率决定了在固定推理预算下能探索多少候选路径。

### Ensemble RM：冗余对抗 Hack

单一RM的任何系统性偏差都可能被策略模型利用。Ensemble RM采用多个异构RM的集成来增强鲁棒性：

- **架构多样性**：混合不同大小、不同初始化的RM（如Llama-based + GPT-based）
- **Prompt多样性**：同一RM使用不同的评分提示（如「评估正确性」vs「评估有用性」）
- **聚合策略**：

$$\bar{r}(y) = \text{Agg}(r\_1(y), r\_2(y), ..., r\_M(y))$$

常见聚合：$\text{Mean}$（均值，保留综合偏好）和 $\text{Min}$（最小值，最保守——只有所有RM都认可的回复才得高分，对Reward Hacking有最强抵抗力）。实践中常使用 $\text{Min} + \text{Mean}$ 的混合策略：$\bar{r} = \gamma \cdot \min\_m r\_m + (1-\gamma) \cdot \text{mean}\_m r\_m$，通过调节 $\gamma$ 在安全性与效率间权衡。

### 与本文各主题的连接

**(a) Bradley-Terry模型是DPO的隐式数据模型**：DPO之所以能跳过显式RM训练，是因为其损失函数直接从BT模型推导而来。DPO的损失函数 $\mathcal{L}\_{\text{DPO}} = -\log \sigma(\beta(\log \frac{\pi\_\theta(y\_w)}{\pi\_{\text{ref}}(y\_w)} - \log \frac{\pi\_\theta(y\_l)}{\pi\_{\text{ref}}(y\_l)}))$ 本质上是用策略模型自身隐式地参数化了奖励函数 $r\_\theta(y) = \beta \log \frac{\pi\_\theta(y)}{\pi\_{\text{ref}}(y)}$——这是BT模型在策略空间的对偶形式。DPO的简洁正是建立在BT偏好假设成立的前提之上。

**(b) RM质量直接决定PPO/GRPO的最终质量——奖励天花板**：策略优化的目标函数是 $\max\_\theta \mathbb{E}\_{y \sim \pi\_\theta}[r\_\phi(y)]$。若 $r\_\phi$ 与真实人类偏好 $r^*$ 存在系统性偏差 $\Delta r(y) = r\_\phi(y) - r^*(y)$，则最优策略将收敛到最大化 $r^* + \Delta r$ 的方向。策略优化算法（PPO/GRPO/DAPO）只能保证收敛到RM眼中的最优，无法超越RM的认知上限——**RM的偏差就是最终策略的偏差上界**。

**(c) PRM是Agentic RL信用分配的关键**：Agentic RL面临的核心难题之一是将稀疏的轨迹级奖励分解为步骤级信号。PRM直接提供步骤级奖励 $r(s\_t)$，将信用分配从「回溯性推断」转化为「前瞻性评估」——每一步执行后立刻获得质量反馈。这大幅缓解了前文Agentic RL章节中分析的长程因果信号衰减问题（$O(\lambda^T)$ 衰减）。PRM与GAE的关系：GAE用时间衰减加权解决信用分配，PRM用步骤级模型预测直接替代衰减——**PRM提供了更准确的 $V(s\_t)$ 先验**，使GAE的TD误差 $\delta\_t$ 的估计方差显著降低。

**(d) RM分布偏移 vs On-Policy Distillation的分布同步**：RL训练中，策略 $\pi\_\theta$ 的分布持续变化，而RM通常是预训练好的固定模型——RM的训练数据来自初始策略的分布。随着训练的推进，RM评估的回复分布与RM训练时见过的分布逐渐偏离，RM的评分质量下降。On-Policy Distillation通过教师模型的持续更新来保持分布同步——这正是RM所需的范式：**在RL训练中迭代微调RM（Online RM Training）**，使其始终适应当前策略的输出分布，是避免训练后期RM信号失效的关键工程实践。

## Policy Optimization

### 从SFT到RL：优化视角的统一

SFT与RL并非两种截然不同的范式，从优化目标出发，二者可以统一在同一个数学框架下理解。

**SFT的本质是行为克隆（Behavior Cloning, BC）**：给定专家数据集 $\mathcal{D} = \{(x, y^*)\}$，SFT通过最大化条件对数似然 $\max\_\theta \mathbb{E}\_{(x,y^*)\sim\mathcal{D}}[\log \pi\_\theta(y^*|x)]$ 来训练策略。这等价于在专家状态分布上进行监督学习——模型只见过专家的"正确路径"，从未被要求评估或比较其他候选回复。**SFT的优势在于训练稳定**（目标函数为凸优化问题，梯度方向明确），但**致命缺陷是分布外（OOD）崩塌**：当推理时遇到训练分布未覆盖的输入模式，模型的行为完全不可控，容易产生幻觉或低质量回复。

**RL的做法是奖励驱动的最优策略搜索**：引入奖励函数 $r(x,y)$（来自Reward Model或规则验证器），优化目标变为 $\max\_\theta \mathbb{E}\_{x\sim\mathcal{D}, y\sim\pi\_\theta(\cdot|x)}[r(x,y)]$。RL允许模型**主动探索**——对同一输入采样多条不同的输出，通过奖励信号区分优劣，再用策略梯度（Policy Gradient）的方式将高奖励输出的概率推高。**重要性采样（Importance Sampling）** 是实现这一目标的核心数学工具：

$$\max\_\theta \mathbb{E}\_{x,y\sim\pi\_{\theta\_{\text{old}}}} \left[ \frac{\pi\_\theta(y|x)}{\pi\_{\theta\_{\text{old}}}(y|x)} \cdot A(x,y) \right]$$

当重要性比率 $\frac{\pi\_\theta}{\pi\_{\theta\_{\text{old}}}} \equiv 1$（即不做任何策略更新，仅在当前策略分布上做加权），RL损失退化为带优势权重的SFT损失。**SFT是RL在"固定分布、无探索"条件下的退化特例**——SFT只在专家数据上拟合（优势恒为1，比率恒为1），而RL在模型自主生成的分布上（比率≠1）利用优势信号区分好坏，打破行为克隆的分布束缚。

**核心对比**：

| 维度 | SFT (MLE / BC) | RL (Policy Optimization) |
|------|---------------|--------------------------|
| 优化目标 | $\max \log\pi(y^*|x)$ | $\max \mathbb{E}[r(x,y)]$ |
| 数据来源 | 固定离线数据集 | 当前策略在线采样 |
| 分布覆盖 | 仅专家分布 | 完整策略分布（含探索） |
| OOD鲁棒性 | 差（崩塌） | 好（奖励提供 guard） |
| 训练稳定性 | 高 | 低（需clip/KL约束） |
| 创新潜力 | 仅模仿已有模式 | 可涌现新策略 |

这一视角解释了为什么在实际训练中通常采用 **SFT → RL 的两阶段流程**：SFT先将策略定位到专家行为附近（提供稳定初始化），RL再在附近进行有界探索以超越专家——**SFT保下限，RL冲上限**。

### PPO（Proximal Policy Optimization）

PPO是RL在LLM对齐中的基础算法，核心思想是：在持续优化策略的同时，通过clip操作严格约束每次更新的幅度，保证训练稳定。

PPO采用Actor-Critic架构，训练中包括四个模型：Actor（策略模型，负责生成回复）、Critic（价值模型，估计期望回报作为优势函数的baseline）、Reward Model（提供逐token或序列级奖励信号）以及Ref Model（冻结的初始策略，提供KL正则化锚点）。

**优势函数** $A\_t$ 的构建是PPO的核心。PPO以Critic模型估计的状态价值 $V(s\_t)$ 为baseline，通过**GAE（广义优势估计）** 计算未来每一步的TD误差 $\delta\_t = r\_t + \gamma V(s\_{t+1}) - V(s\_t)$，然后进行指数衰减的加权求和：

$$A\_t^{\text{GAE}(\gamma, \lambda)} = \sum\_{l=0}^{\infty} (\gamma\lambda)^l \delta\_{t+l}$$

其中 $\gamma$ 控制远期奖励的衰减，$\lambda$ 在偏差（$\lambda=0$，仅用单步TD误差）和方差（$\lambda=1$，用完整蒙特卡洛回报）之间做 bias-variance trade-off。此外，每一步的奖励中会加入与Ref模型的**KL散度惩罚项** $-\beta \cdot \text{KL}(\pi\_\theta \parallel \pi\_{\text{ref}})$，防止Actor在探索过程中偏离初始策略太远。

PPO的关键操作为**clip机制**。记重要性比率 $r\_t(\theta) = \frac{\pi\_\theta(a\_t|s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t|s\_t)}$，PPO的目标函数为：

$$\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}\_t \left[ \min\left( r\_t(\theta) A\_t,\ \text{clip}(r\_t(\theta), 1-\epsilon, 1+\epsilon) A\_t \right) \right]$$

clip将比率强行限制在 $[1-\epsilon, 1+\epsilon]$ 内（典型 $\epsilon=0.2$），$\min$ 操作保证当 $A\_t>0$ 时不因过度增大比率而获得虚假收益，当 $A\_t<0$ 时不因过度减小比率而逃避惩罚。这从根本上**防止策略在单步更新中变化过猛导致策略崩溃（policy collapse）**——一旦新旧策略差异过大，clip直接截断梯度，模型在该样本上停止更新。
### DPO（Direct Policy Optimization）
DPO的核心思想是绕过传统的reward model和复杂的PPO强化学习阶段，直接使用人类偏好数据对语言模型进行优化。

它通过损失函数拉大模型生成chosen回复和rejected回复的概率差，避免了PPO训练时的在线采样成本以及Actor-Critic架构的崩溃风险。

```python
def dpo\_loss(chosen\_logp, rejected\_logp, chosen\_ref\_logp, rejected\_ref\_logp, beta=0.1):
    chosen\_logratio = chosen\_logp-chosen\_ref\_logp
    rejected\_logratio = rejected\_logp-rejected\_ref\_logp
    logratio = chosen\_logratio-rejected\_logratio
    loss = -F.logsigmoid(beta*logratio)
    return loss.mean()
```

### GRPO（Group Relative Policy Optimization）

#### 统一的变体分析框架

PPO、GRPO、DAPO、GSPO等算法并非彼此孤立，而是基于**同一套数学积木**在不同维度上的差异化组合。统一框架的核心组件为：

$$\mathcal{L} = \mathbb{E} \left[ \min\left( \underbrace{r(\theta)}\_{\text{重要性比率}} \cdot \underbrace{A\_i}\_{\text{优势}},\ \underbrace{\text{clip}(r(\theta))}\_{\text{截断比率}} \cdot A\_i \right) - \beta \cdot \underbrace{D\_{\text{KL}}}\_{\text{KL惩罚}} \right]$$

所有变体共享这三大积木，区别仅在三个维度：(1) **优势计算**——用Critic的GAE估计还是组内标准化；(2) **损失聚合粒度**——token级、序列级还是batch级求平均；(3) **clip参数**——上下界对称还是非对称，clip加在哪一层。下表总结了核心差异：

| 维度 | PPO | GRPO | DAPO | GSPO |
|------|-----|------|------|------|
| 优势 $A\_i$ | Critic + GAE | 组内Z-score标准化 | 组内Z-score标准化 | 组内Z-score + 累积奖励 |
| 损失聚合 | Token级平均 | Token级平均 | Token级直接聚合 | **序列级聚合** |
| Clip 参数 | $\epsilon=0.2$ 对称 | $\epsilon=0.2$ 对称 | **上界放大** (higher) | $\epsilon$ 对称，序列级 |
| Critic | ✅ 需要 | ❌ 不需要 | ❌ 不需要 | ❌ 不需要 |

#### 为什么GRPO取代PPO：Critic在LLM中的三大困境

GRPO的核心创新是**废除Critic模型**，直接用组内相对排名计算优势。这一设计的背后是Critic在LLM场景中的三个致命问题：

**1. 高方差（High Variance）**：LLM的生成空间极为庞大（每步从数万词表中采样），状态价值函数 $V(s)$ 的估计天然具有高方差。PPO的GAE是MC和TD的折中方案，但在长序列（数百至数千token）中，$\lambda$ 的 bias-variance trade-off 极难调参——$\lambda$ 偏高则方差爆炸，$\lambda$ 偏低则偏差让训练信号失去区分度。

**2. 额外容量开销（Extra Capacity）**：一个与Actor（动辄几十B参数）同规模的价值网络意味着近乎翻倍的显存和计算开销。在LLM RL训练中，Actor本身已占绝大部分GPU资源，再加一个等大小的Critic严重限制了batch size和序列长度。

**3. 参数共享冲突（Parameter Sharing Conflict）**：实践中常让Actor和Critic共享底层transformer参数以节省显存（仅顶层head不同）。但生成任务（next-token prediction）和价值估计（scalar regression）的优化方向相互矛盾——生成需要保留丰富语义细节，价值估计需要将语义压缩为单一标量。这种冲突会同时损害生成质量和价值估计的准确性。

GRPO用一个巧妙的统计替代方案解决了这三个问题：**对同一prompt采样一组（Group）$G$ 条回复，用组内均值与标准差做Z-score标准化获得相对优势**：

$$A\_i = \frac{r\_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \varepsilon}$$

其中 $\mathbf{r} = \{r\_1, r\_2, ..., r\_G\}$ 为组内$G$条回复的奖励得分。这一做法之所以有效，是因为**组内相对排序提供了一种无需估计绝对价值的"胜者通吃"信号**——在给定prompt下，模型只需知道哪些回复比其他更好，无需知道"多好"。统计学上，Z-score标准化退化为大数定律下的秩检验，当组大小$G$足够大时（通常$G=4\sim16$），组内标准化后的相对优势在期望上等价于真实优势的单调变换。

GRPO的损失函数为：

```python
def grpo\_loss(rewards, logp\_per\_token, ref\_logp\_per\_token, old\_logp\_per\_token, mask=None, beta=0.01, clip\_eps=0.2):
    mean\_rewards = rewards.mean(dim=-1, keepdim=True)
    std\_rewards = rewards.std(dim=-1, keepdim=True)
    advantages = (rewards - mean\_rewards) / (std\_rewards + 1e-8)
    advantages\_per\_token = advantages.unsqueeze(-1)

    ratio = torch.exp(logp\_per\_token-old\_logp\_per\_token)
    adv1 = ratio*advantages\_per\_token
    adv2 = torch.clamp(ratio, 1-clip\_eps, 1+clip\_eps)*advantages\_per\_token

    kl\_per\_token = torch.exp(ref\_logp\_token-logp\_per\_token)-(ref\_logp\_token-logp\_per\_token)-1

    loss\_per\_token = -(torch.min(adv1, adv2) - beta*kl\_per\_token)
    policy\_loss = (loss\_per\_token*mask).sum(dim=-1)/mask.sum(dim=-1)
    return policy\_loss.mean()
```

其中KL散度采用 **k3 估计器** $\text{KL} = \exp(z\_\theta - z\_{\text{ref}}) - (z\_\theta - z\_{\text{ref}}) - 1$（其中 $z = \log\pi$），这是KL散度的一个二阶泰勒展开近似形式，在 $z\_\theta \approx z\_{\text{ref}}$ 时精确成立。相比直接计算 $\log\frac{\pi\_\theta}{\pi\_{\text{ref}}}$，k3估计器在数值上更稳定，且天然非负（$f(x)=\exp x - x - 1 \geq 0$ 当取等时 $x=0$）。

### DAPO（Decoupled clip and Dynamic sAmpling Policy Optimization）

DAPO在GRPO的基础上做出四项关键改进，每项改进都有明确的理论矛头所向：

**1. Higher Clip（上界放宽）**：GRPO采用对称clip $[1-\epsilon, 1+\epsilon]$，这对于正优势（$A\_i>0$）和负优势（$A\_i<0$）施加了对称的更新幅度限制。但问题在于：正优势样本是模型需要**强化**的高质量输出，负优势样本是需要**抑制**的低质量输出。对称clip下，一个奖励极高的样本（比率>>1+ε）会被截断——这意味着模型失去了最大化利用好样本的机会，策略多样性被不必要地压缩。DAPO将上界放大（如 $\epsilon\_{\text{upper}} > \epsilon\_{\text{lower}}$），允许模型对高奖励输出做更大步的强化更新，同时保持下界限制以防止过度惩罚。数学上体现为非对称clip：

$$\text{clip}(r(\theta)) = \min\left(r(\theta),\ 1+\epsilon\_{\text{upper}}\right) \quad \text{而非} \quad \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)$$

**2. Dynamic Sampling（动态采样）**：GRPO的组内标准化依赖样本方差提供区分度——若组内所有回复的奖励相同（$\text{std}(\mathbf{r}) \approx 0$），则 Z-score 分母趋近于零，优势信号失效。这种情况在遇到极简单（所有回复全对）或极困难（全错）的prompt时频繁出现，称为**零方差问题（Zero-Variance Problem）**。DAPO的动态采样强制要求每组样本的奖励方差超过最低阈值：在采样阶段持续生成候选回复，过滤掉奖励完全一致的组，或在组内通过拒绝采样补充差异化样本，确保 $\text{std}(\mathbf{r}) > \tau\_{\text{min}}$。理论上这等价于为策略梯度注入了有信息量的比较信号的最小充分条件。

**3. Token-Level Policy Gradient Loss（Token级直接聚合）**：这是四项改进中**理论动机最深**的一项。GRPO在序列级别计算平均loss：

$$\mathcal{L}\_{\text{seq}} = \frac{1}{N} \sum\_{n=1}^{N} \frac{1}{T\_n} \sum\_{t=1}^{T\_n} \ell\_{n,t}$$

其中 $N$ 为组内样本数，$T\_n$ 为第 $n$ 个样本的token数。这种做法的隐蔽缺陷称为**长序列梯度稀释（Long-Sequence Gradient Dilution）**：对于一条高质量但很长的推理链（如包含反思和自我纠错的2000-token输出），每个token对梯度的平均贡献仅为 $1/2000$；而一条短回复（如直接给出答案的50-token输出）的token级贡献权重高达 $1/50$。这意味着**长推理链中的关键逻辑步骤会在反向传播中被系统性地低估**，模型难以学到高阶推理模式。

DAPO将聚合方式改为Token级直接求和后再做全局归一化：

$$\mathcal{L}\_{\text{DAPO}} = \frac{\sum\_{n,t} \ell\_{n,t} \cdot \mathbb{I}[\text{mask}\_{n,t}]}{\sum\_{n,t} \mathbb{I}[\text{mask}\_{n,t}]}$$

这确保每个有效token对梯度的贡献完全均等，消除了序列长度对学习信号强度的扭曲。对推理训练尤为关键——反思token（"Wait, this is wrong..."）和关键推导步骤获得了与其信息量匹配的梯度权重。

**4. Overlong Reward Shaping（超长惩罚塑形）**：RL训练中，模型会探索出"通过输出更长回复来刷分"的reward hacking行为——验证器可能对更长的回复给出虚假高分（例如包含更多正确片段，即使整体逻辑混乱）。DAPO的做法是：(a) **硬截断**：在损失计算中直接剔除被截断的样本（在生成阶段设置max\_new\_tokens，超出即truncate并在loss中mask掉全部token），避免truncation噪声污染梯度；(b) **软惩罚**：对未截断但较长的回复，引入阶梯式长度惩罚 $r\_{\text{length}} = r \cdot \mathbb{I}[L < L\_{\text{thresh}}] + r \cdot \alpha^{\lfloor L / \Delta L \rfloor} \cdot \mathbb{I}[L \ge L\_{\text{thresh}}]$，鼓励模型在完成任务的前提下尽量精简。这本质是在准确性和经济性之间引入了显式可调的trade-off参数。

### GSPO（Group Sequence Policy Optimization）

GSPO与GRPO的核心分歧在于**重要性采样比率的计算粒度**，这一分歧深刻影响着算法对两类RL场景的适应性。

**Token级比率 vs 序列级比率**：GRPO在token级别计算比率 $r\_t = \frac{\pi\_\theta(y\_t|x, y\_{<t})}{\pi\_{\text{old}}(y\_t|x, y\_{<t})}$，这意味着策略更新被视为**每个token的独立决策**——如果序列前半段与旧策略一致（$r\_t \approx 1$），后半段偏离（$r\_t \gg 1$），前半段token几乎不受益/受罚。这适用于**短程决策**场景（如单轮QA的回复措辞调整），此时token之间的依赖相对局部。

GSPO改为在**序列级别**计算重要性比率：

$$r\_{\text{seq}}(\theta) = \frac{\prod\_{t=1}^{T} \pi\_\theta(y\_t|x, y\_{<t})}{\prod\_{t=1}^{T} \pi\_{\text{old}}(y\_t|x, y\_{<t})} = \exp\left( \sum\_{t=1}^{T} \log \pi\_\theta(y\_t|x, y\_{<t}) - \log \pi\_{\text{old}}(y\_t|x, y\_{<t}) \right)$$

**序列级比率的物理学直觉**：一条轨迹中的所有token共享同一个比率因子——要么整条轨迹被整体强化，要么被整体抑制。这抓住了Agentic场景的本质特征：工具调用序列中的每一步的价值依赖于整条轨迹的最终成功率，单步无法被独立评估。一条最终成功的轨迹，即使其中某一步看起来"异常"，也应当被整体奖励（因为整体成功证明了该步骤在上下文中的合理性）；反之，一条失败的轨迹，即使大部分步骤看起来正常，也应当被整体抑制。

形式化地，GSPO的损失函数为：

$$\mathcal{L}^{\text{GSPO}} = \min\left( r\_{\text{seq}} A\_{\text{seq}},\ \text{clip}(r\_{\text{seq}}, 1-\epsilon, 1+\epsilon) A\_{\text{seq}} \right) - \beta D\_{\text{KL}}$$

其中 $A\_{\text{seq}}$ 为序列级优势（同样通过组内标准化获得），clip在序列级施加。这意味着**clip保护的是整条策略轨迹不被单步大幅修改**——当一条轨迹的累积比率超出 $[1-\epsilon, 1+\epsilon]$，整条轨迹的梯度被截断，模型在该轨迹上停止更新。这比token级clip更保守，但在agentic场景（该场景下轨迹级稳定性比token级灵活性更重要）中更安全。

**适用场景对比**：

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| 单轮QA / 代码生成 | GRPO | Token级灵活性有利于调整局部措辞 |
| 多步推理（CoT） | DAPO | Token级聚合 + 长序列不稀释 |
| Agentic / 工具调用 | GSPO | 序列级比率保护轨迹整体一致性 |
| 通用对齐（多场景混合） | DAPO + GSPO 混合 | 根据任务类型自适应选择 |

## MDP视角下的LLM训练

强化学习在LLM中的应用可以从**马尔可夫决策过程（MDP）**的视角进行统一分析。根据状态转移的特性，可以将LLM的RL训练分为两种基本范式：**Contextual Bandit（上下文赌博机）**和**Full MDP（完整马尔可夫决策过程）**。这一区分不仅是理论上的分类学练习，更直接决定了算法设计、训练稳定性和最终性能上限。

### Contextual Bandit：传统RLHF的数学本质

传统RLHF（包括PPO/DPO/GRPO等算法在标准问答场景中的应用）本质上建模为一个**Contextual Bandit问题**：

- **状态（State）**：$s = \text{prompt}$，即用户输入的提示词。状态在动作执行前后不发生改变。
- **动作（Action）**：$a = \text{full response}$，即模型生成的完整回复序列。
- **轨迹长度（Trajectory Length）**：$T = 1$，单步决策，不存在状态转移。
- **状态转移函数**：$P(s'|s, a) = \delta(s' = s)$，退化为恒等映射——无论模型生成什么回复，都不会改变当前"对话上下文"，因为下一轮对话在训练中被视为一个全新的独立episode。
- **奖励（Reward）**：$r(s, a)$，由Reward Model或规则验证器对完整回复打分。

从数学上看，Contextual Bandit是MDP在转移函数退化为恒等映射时的特例。其Bellman方程退化为：

$$Q(s, a) = r(s, a)$$

价值函数等于即时奖励——因为不存在未来状态，也就没有"未来期望回报"的概念。策略梯度的形式也大幅简化：

$$\nabla\_\theta J(\theta) = \mathbb{E}\_{s \sim \mathcal{D}, a \sim \pi\_\theta(\cdot|s)}\left[\nabla\_\theta \log \pi\_\theta(a|s) \cdot A(s, a)\right]$$

其中优势函数 $A(s, a)$ 仅依赖于当前状态-动作对的即时奖励，无需考虑长期回报的折扣累积。这一简化带来了训练上的巨大便利——梯度估计的方差仅来自单步动作采样，不涉及时间维度的误差传播。

**Contextual Bandit何时足够？** 当任务满足以下条件时，CB建模完全充分：(1) 任务是单轮的，输入和输出之间没有交互循环；(2) 回复质量可以通过最终输出直接评估，无需中间步骤的验证；(3) 不存在工具调用或环境反馈改变后续决策的需求。典型场景包括：单轮问答质量对齐、文本摘要偏好优化、有害内容拒绝等。在这些场景下，PPO/GRPO/DAPO等算法在CB框架内已经表现优异——前文分析的GRPO取代PPO的三大理由（Critic高方差、额外容量开销、参数共享冲突）正是在CB设定下成立的。

**Contextual Bandit何时失效？** CB在以下场景中暴露出根本性的建模不足：

1. **多步工具调用**：当LLM需要先调用搜索引擎获取信息，再基于结果生成回答时，第一次工具调用的结果改变了后续决策的"状态"，这不再是单步决策。CB若强行应用，只能将"prompt + 搜索结果"视为新的单一prompt，但这丢失了"主动选择调用什么工具、如何解读结果"的决策学习。

2. **代码调试循环**：LLM编写代码→执行→观察错误→修改代码→再执行，每次执行结果都是新的状态信息。CB无法建模这种"观察→调整→再观察"的闭环，因为每次工具调用都被拍扁为单步回复的一部分。

3. **长期规划与任务分解**：如"帮我预订机票并安排行程"，涉及多个子任务的串联执行，每个子任务的成功与否影响后续决策路径。CB无法学习到"根据中间结果动态调整计划"的元能力。

在这些场景下强行用CB建模，相当于将多步交互压缩为单步输出，损失了中间反馈提供的丰富学习信号，且无法学习到"根据中间结果调整策略"的元能力——而这恰恰是Agent能力的核心。

### Full MDP：Agentic RL的数学框架

Agentic RL将LLM与环境的交互建模为完整的**有限视界MDP**：

- **状态（State）**：$s\_t = (\text{prompt}, h\_t, o\_t)$，包括原始prompt、工具调用历史 $h\_t = (a\_1, r\_1, a\_2, r\_2, ..., a\_{t-1}, r\_{t-1})$ 以及当前环境观测 $o\_t$（如代码执行结果、网页内容、API返回值）。状态的维度随 $t$ 增长而持续膨胀——这是MDP场景下Critic训练困难的根源之一。

- **动作（Action）**：$a\_t \in \mathcal{A}$，可以是生成文本、调用工具、终止任务等。动作空间 $\mathcal{A}$ 是语言token空间与结构化tool call空间的并集，具有"离散（工具选择）+ 连续（参数生成）"的混合特性。

- **状态转移（Transition）**：$P(s\_{t+1}|s\_t, a\_t)$ 由两部分组成——确定性部分（环境对tool call的响应，如代码执行结果是确定性的）和随机性部分（外部API的不确定性、网络延迟、网页内容的动态变化等）。

- **奖励（Reward）**：$r\_t = r(s\_t, a\_t)$ 或仅在终止状态给出 $r\_T$（稀疏奖励）。常见形式包括 $r\_T = \mathbb{I}[\text{任务成功}]$（二元）或 $r\_T = \text{score}$（连续）。

- **轨迹（Trajectory）**：$\tau = (s\_0, a\_0, s\_1, a\_1, ..., s\_T)$，长度 $T$ 不定，取决于任务复杂度和模型策略（何时选择"终止"动作）。

在Full MDP下，策略优化的目标变为最大化累积折扣奖励的期望：

$$J(\theta) = \mathbb{E}\_{\tau \sim \pi\_\theta}\left[\sum\_{t=0}^{T} \gamma^t r\_t\right]$$

其中 $\gamma \in (0, 1]$ 是折扣因子，控制短期与长期回报的权衡。与CB相比，Full MDP的核心复杂性来自**时间维度的引入**——每一步的决策不仅影响即时奖励，还通过状态转移影响未来的所有决策机会。这也意味着**策略更新不再独立于轨迹**：更新第 $t$ 步的动作概率，会通过改变后续状态的分布间接影响所有 $k > t$ 步的优化。

### 从CB到MDP：算法设计的连锁反应

建模视角从CB升级到MDP，直接导致策略优化算法的设计面临一系列根本性挑战。理解这些挑战是评估PPO/GRPO/DAPO/GSPO在Agentic场景下适用性的关键。

**1. PPO Clip为何在CB中高效、在MDP中受限？**

在Contextual Bandit设定下，PPO的clip操作之所以高效，是因为单步决策不存在时间维度的误差累积：
- 重要性采样比率 $r\_t(\theta) = \frac{\pi\_\theta(a|s)}{\pi\_{\theta\_{\text{old}}}(a|s)}$ 仅涉及当前步的策略比值。
- Clip将其限制在 $[1-\epsilon, 1+\epsilon]$ 后，对整体轨迹的偏差影响是**局部的、可控的**。
- 没有"一步错、步步错"的连锁反应——即使某次更新过于激进，也仅影响当前回复质量，不会通过状态转移"污染"后续决策。

在Full MDP下，clip的保护作用被**时间维度稀释**：
- 即使每步的ratio被clip在 $[1-\epsilon, 1+\epsilon]$ 内，$T$ 步累积的策略偏移可达 $(1+\epsilon)^T$——随轨迹长度指数增长。换言之，**单步clip无法阻止多步累积的策略漂移**。
- 优势函数 $\hat{A}\_t$ 的估计需要累积未来奖励，方差随轨迹长度增长（见下文信用分配分析），clip后的梯度信号中信噪比进一步恶化。
- PPO在MDP下的有效使用要求额外的稳定化手段：截断轨迹长度、增大batch size以降低方差、更保守的clip阈值（$\epsilon < 0.1$）。

**2. 信用分配（Credit Assignment）的维度爆炸**

从CB切换到MDP后，信用分配问题从"单步归因"升级为"多步因果推断"：

- 一条Agentic轨迹可能包含20-50步工具调用，最终奖励（任务成功/失败）需要在所有中间步骤间分配"功劳"或"责任"。
- **REINFORCE估计器** $\nabla\_\theta J = \mathbb{E}[\sum\_t \nabla\_\theta \log \pi\_\theta(a\_t|s\_t) \cdot G\_t]$ 中，回报 $G\_t = \sum\_{k=t}^T \gamma^{k-t} r\_k$ 的方差随 $T$ 线性增长——这是著名的"信用分配方差问题"，在MDP中尤为严重。
- **GAE（广义优势估计）**通过引入Critic的 $V(s\_t)$ 作为baseline并在时间维度上做 $\lambda$ 指数衰减加权，将方差从 $O(T)$ 压缩至 $O(T \cdot \lambda^T)$。但当 $T$ 较大时（如Agentic场景下 $T=20+$），即使 $\lambda=0.95$，衰减因子 $\lambda^{20} \approx 0.36$——长距离的因果信号仍然无法有效传播。

更深层的问题在于**时间维度的因果解耦**：假设一条代码修复轨迹包含20步操作，第5步引入了一个"隐式bug"（修改了配置文件），第15步才因该bug导致测试失败。最终奖励信号反馈到第5步需要跨越10步的梯度传播，而在这10步中间，模型又做了大量"正确"的操作——这些操作的梯度信号被**错误地"污染"**：既承担了失败的责任（因为最终失败了），又被鼓励（因为中间步骤本身是正确的）。标准GAE的指数衰减无法区分这种非马尔可夫的因果结构。

**3. GRPO组内相对优势的MDP适配**

GRPO的核心创新——用组内标准化替代Critic的价值估计——在MDP场景下需要仔细重新设计：

- **轨迹级优势（最直接）**：对同一prompt，采样 $G$ 条完整轨迹 $\tau\_1, ..., \tau\_G$，计算每条轨迹的总奖励 $R(\tau\_i) = \sum\_t r\_t^{(i)}$，然后进行组内标准化 $A(\tau\_i) = \frac{R(\tau\_i) - \text{mean}(R)}{\text{std}(R)}$。将轨迹级优势均匀分配给轨迹内的每一步。优点是实现简单；缺点是丢失了步级区分度——轨迹中每一步都被赋予相同优势，信用分配的粒度粗糙。

- **步级优势（更精细）**：在轨迹的每个时间步 $t$，对 $G$ 条轨迹中第 $t$ 步的动作进行组内比较。这需要**轨迹对齐**（不同轨迹长度不同，需要确定哪些步之间具有可比性），实现上更复杂但信用分配更精确。

- **帧级优势**：对轨迹中每个tool call的生成token进行细粒度优势计算，类似DAPO的token-level损失在MDP下的推广。这需要对tool call中的关键参数token（如文件路径、命令参数）和模板token（如JSON格式token）施加差异化权重。

**GRPO在MDP下的核心难题——组内方差来源混杂**：在CB场景下，组内方差几乎完全来自"回复质量差异"，Z-score标准化能准确捕捉相对优劣。但在MDP下，组内轨迹的方差来源包括三层：(a) **策略多样性的合理差异**——两条轨迹可能都成功，但走了完全不同的工具调用路径；(b) **环境随机性的不可控差异**——同一工具调用在不同时间可能返回不同结果（如API限流、网页内容更新）；(c) **真实质量差异**——一条轨迹比另一条在完成任务上确实更好。组内标准化会将(a)和(b)的方差错误地映射为"质量差异"，将合理的路径多样性当作错误来惩罚。这是GRPO直接应用于Agentic轨迹时最棘手的理论障碍。

**4. 探索（Exploration）需求的质变**

CB场景下的探索相对简单：对同一prompt采样多样化的回复即可，策略熵正则化（entropy bonus）足以保证足够的多样性。但在MDP下，探索的需求发生**质变**：

- **时序探索（Temporal Exploration）**：不仅要在每个决策点做出多样化的动作选择，还要探索不同的**动作序列**。动作序列的组合空间随轨迹长度呈**指数增长**——若有 $K$ 种工具，轨迹长度 $T$，则可能的工具调用序列数为 $K^T$。在 $K=10, T=20$ 的典型Agentic场景下，组合空间高达 $10^{20}$，远超任何采样预算的覆盖范围。
- **工具组合探索（Tool Composition Exploration）**：Agent需要发现"先用搜索获取信息，再用代码分析数据，最后用文件写入保存结果"这样的有效工具组合模式。这需要对工具间的**因果关系**进行系统性探索——不是随机尝试工具组合，而是理解哪些工具组合在语义上是连贯的。
- **探索-利用的跨步平衡（Stepwise Explore-Exploit Balance）**：早期步骤应偏向探索（收集环境信息，为后续决策奠定基础），后期步骤应偏向利用（基于已收集信息做最优决策）。这需要**时间感知的探索策略**，而非全局恒定的探索率（如固定的temperature）。

当前Agentic RL的探索主要依赖简单的温度采样和动作空间噪声，远不足以应对组合工具的复杂性。更系统的探索方法——如基于好奇心（Curiosity-driven）的内在奖励探索、基于状态新颖性（Novelty-based）的访问计数、基于熵最大化（MaxEnt RL）的策略正则化——是Agentic RL从实验走向实用的关键瓶颈。

---

## Agentic RL

Agentic RL 是强化学习在 LLM 领域的重要延伸：让模型通过与交互式环境的多轮交互，自主学会使用工具、规划步骤、完成复杂任务。与标准 RLHF 的核心区别在于，RLHF 关注的是单轮问答的回复质量（Contextual Bandit），而 Agentic RL 关注的是跨越多个步骤的工具调用序列（Full MDP）——模型的每一次动作（写代码、调用 API、浏览网页）都会改变环境状态，并影响后续决策。

从MDP视角来看（详见上一节），Agentic RL将LLM的训练从Contextual Bandit升级为Full MDP，这带来了维度更高、时序更长的优化难题。本节深入分析这些挑战，并讨论上文策略优化算法家族在Agentic场景下的适用性与局限。

### 关键组件
- **环境（Environment）**：代码执行沙箱（如 Docker 容器）、浏览器模拟器、数据库查询接口或第三方 API 等，负责接收 LLM 的动作并返回观测结果。环境的**确定性程度**直接影响训练难度——完全确定性的代码执行环境比带有网络延迟和随机性的Web环境更容易学习。环境的**可复现性**也是关键工程挑战：外部API和网页内容的变化使同一prompt在不同时间的执行结果可能不同，破坏了RL训练的状态一致性假设。
- **动作空间（Action Space）**：以 tool call 为基本单元，包括文件读写、代码执行、搜索检索、终端命令等结构化操作。动作空间的离散化设计直接影响探索效率。关键设计选择包括：(1) **扁平化 vs 层次化**——将所有tool call参数展开为扁平token序列 vs 先用高层策略选择工具再用低层策略填充参数；(2) **开放参数 vs 约束参数**——代码内容属于开放生成，文件路径属于约束选择，两者需要不同的参数化策略。
- **奖励设计（Reward Design）**：通常分为结果奖励（任务是否最终完成）和过程奖励（中间步骤的正确性）。结果奖励信号稀疏但对齐目标更准确；过程奖励需要额外的 Process Reward Model（PRM），但能提供更密集的学习信号。实际系统中通常采用**混合奖励**：结果奖励保证目标对齐，过程奖励提供中间监督，外加格式奖励（tool call JSON合法性）和长度惩罚（控制轨迹长度）作为正则化。

### 代表性工作
**SWE-bench** 是衡量 LLM 自动修复 GitHub Issue 能力的基准，其配套的 **SWE-agent** 框架定义了一套标准的代码操作 tool call 接口，是 Agentic RL 代码领域的重要试验床。**WebArena** 则聚焦于浏览器操作任务，提供网页交互的真实环境评估。**ToolRL** 等工作进一步探索了将 GRPO 类算法直接应用于工具调用轨迹的训练范式。**OSWorld**（Xu et al., 2024）提供了跨多应用（OS-level）的Agent评估基准，涉及文件管理、网页浏览、代码执行等多工具组合场景。**AgentGym**（Xi et al., 2024）构建了涵盖Web、Shell、数据库等多种环境的Agent训练与评测平台，支持行为克隆与RL的混合训练。

### 核心挑战的深层分析

#### 1. 稀疏奖励与探索困境

Agentic RL中的稀疏奖励不仅表现为"信号只在最后出现"，更具有以下深层特性：

- **二元奖励面的极度不平滑**：大多数Agentic benchmark的评估指标是二元的（通过/失败），奖励面（reward landscape）极度不平滑——大量轨迹得到0奖励，少数轨迹得到1，梯度信号在奖励面的大部分区域近乎为零。模型从SFT初始化出发，对工具调用的成功率极低（通常<5%），完全随机探索可能需要数万条轨迹才能发现第一个有效成功样本——远超当前RL训练的覆盖范围（通常几千步）。

- **捷径策略陷阱（Shortcut Trap）**：在奖励稀疏的环境下，模型可能收敛到"不调用任何工具、直接猜测答案"的退化策略。虽然成功率极低，但从策略熵的角度看是最"安全"的——不需要承担工具调用的失败风险，每次输出至少是一个"合法"的回复。RL的梯度信号倾向于首先强化这种低风险策略，使其成为难以逃脱的局部最优。

- **冷启动的示范依赖**：由于从零探索的成本过高，当前Agentic RL训练几乎都依赖**示范引导（Demonstration Bootstrap）**——用少量高质量的人类或强模型示范轨迹初始化replay buffer，为RL提供"第一个正确方向"。这本质上是一种隐式的课程学习：先通过行为克隆将策略拉到成功率>10%的区域，再让RL在奖励信号足够密集的区域进行精细优化。

缓解策略：(1) **课程学习（Curriculum Learning）**——从简单任务（如单步文件读取）逐步过渡到复杂任务（如多文件联合调试），确保早期有足够的正向奖励；(2) **内在奖励（Intrinsic Reward）**——基于状态新颖性（novelty）或预测误差（prediction error）的探索奖励，鼓励模型访问未见过的高信息量状态；(3) **Hindsight Relabeling**——对失败轨迹，回溯分析哪一步如果做了不同选择可能成功，重新标注该步的"反事实奖励"，在关键决策点注入低方差的训练信号。

#### 2. 信用分配的技术深水区

信用分配是Agentic RL中最具挑战性的理论问题，其难度远超标准RLHF场景：

**非马尔可夫的因果结构**：工具调用之间的依赖不是简单的马尔可夫链，而是有向无环图（DAG）——"先搜索再读文档再写代码"，搜索的结果影响文档选择，文档选择影响代码逻辑。这种非线性依赖使得基于时间衰减的标准信用分配（如GAE中的 $\lambda$ 衰减）无法准确捕获跨步骤的因果结构。例如，第3步的搜索质量决定了第7步的文档选择质量，进而决定了第15步的代码质量——但第3步与第15步之间的因果关系跨越了12步中间操作，GAE的指数衰减使这层因果信号衰减至 $\lambda^{12}$，几乎不可见。

**算法层面的连锁后果**：

- **PPO中优势估计的方差爆炸**：在MDP下，优势函数 $\hat{A}\_t$ 的方差随轨迹长度呈 $O(T)$ 增长（REINFORCE）或 $O(T \cdot \lambda^T)$（GAE with $\lambda < 1$）。对于 $T=30$ 的Agentic轨迹，即使 $\lambda=0.95$，前期步骤的优势估计几乎退化为噪声——信噪比过低以至于无法驱动有意义的梯度更新。

- **GRPO的组内方差信号退化**：如前文MDP适配分析所述，MDP下组内轨迹的方差只有一小部分来自真实质量差异。在Agentic场景中，两条轨迹的最终奖励不同，可能不是因为策略优劣，而是因为环境在不同时刻的行为不同（如API返回了不同的搜索结果）。组内Z-score标准化无法区分这种"环境噪声导致的奖励差异"与"策略质量导致的奖励差异"，优势信号的区分度大幅下降。

- **DAPO的token-level损失在MDP下的新意义**：DAPO的token-level损失设计初衷是解决长推理链的梯度稀释问题。在Agentic MDP下，这一问题以新的形式出现——tool call中的关键参数token（如文件路径、命令参数）对任务成败的影响远大于模板token（如JSON格式token），但标准聚合将它们一视同仁。**Token重要性加权**（对tool call的关键参数token施加更高loss权重）是DAPO思路在MDP下的自然延伸，但这需要引入额外的token重要性估计机制（如基于注意力权重或梯度的token saliency检测）。

#### 3. 动作空间的结构化挑战

Agentic RL的动作空间具有"离散连续混合"的特性——工具选择是离散的（搜索 vs 代码执行 vs 文件读取），参数生成是条件连续的（代码内容属于开放生成，文件路径属于约束选择）。这种混合空间使标准的策略参数化面临两难：

- **统一token化**（如SWE-agent的做法）：将整个tool call序列化为特定格式的token流，利用LLM自回归的能力统一处理。优点是实现简单，与现有RL训练流程兼容；缺点是丢失了动作的结构化先验——模型需要从零学习"工具名称token后应跟随参数token"这种语法规则，而非直接施加结构约束。
- **层次化策略**：引入meta-controller（选工具）+ sub-controller（填参数）的两层架构。优点是利用了工具选择的离散性和参数生成的条件性作为归纳偏置（inductive bias）；缺点是增加了训练复杂度，两层策略的联合优化需要处理非平稳的奖励分配问题。

#### 4. 环境交互的成本与安全壁垒

与RLHF中Reward Model的单次推理不同，Agentic RL的每次rollout需要真实的代码执行、网页浏览或API调用。成本壁垒体现在：(1) **延迟放大**——一个20步轨迹的环境交互时间可能是单次推理的100倍以上，严重影响RL训练的吞吐量；(2) **安全风险**——代码执行需要严格的沙箱隔离（Docker容器 + 资源限制 + 网络白名单），任何沙箱逃逸都可能导致严重的安全事故；(3) **可复现性危机**——外部API和网页内容的变化使同一prompt在不同时间的执行结果不可复现，破坏了RL训练的状态一致性假设，导致训练曲线的不确定性显著增加。

### 与策略优化算法的连接

回顾上文讨论的策略优化算法家族，它们在Agentic RL（Full MDP）场景下的定位与局限如下：

| 算法 | CB场景定位 | MDP场景适用性 | 核心局限 |
|------|-----------|-------------|---------|
| **PPO** | 标杆算法，clip保证单步稳定更新 | 需配合GAE+Critic，但Critic对高维文本状态拟合困难 | 长轨迹下优势方差 $O(T)$，Critic训练不稳定 |
| **DPO** | 离线偏好优化，无需在线采样 | 不直接适用——DPO假设单轮 $(y\_w, y\_l)$ 偏好对，无法建模多步轨迹 | 无法处理状态转移，偏好数据的构造在MDP下定义模糊 |
| **GRPO** | 用组内标准化替代Critic，高效稳定 | 轨迹级组内比较可行，但组内方差来源混杂 | 路径多样性与质量差异的混淆；步级信用分配粗糙 |
| **DAPO** | Token-level损失 + higher clip + 动态采样 | Token重要性加权对tool call关键参数有意义；动态采样对抗MDP的零方差问题更关键 | 需要额外的token重要性估计机制 |
| **GSPO** | 序列级重要性采样比率 | 序列级clip对完整Agentic轨迹更自然——一条轨迹的整体质量比单步比率更有意义 | 轨迹长度不一致时实现复杂；序列级聚合可能掩盖关键步的信号 |

**当前前沿方向**：

1. **Hybrid Critic设计**：对Agentic状态的可观测部分（环境观测、工具调用结果）使用结构化编码器（如code-specific encoder或AST-based encoder）构建低维状态表示，再输入Critic做价值估计，缓解高维文本状态的Critic拟合困难。这是让PPO在Agentic MDP下重获竞争力的关键路径。

2. **时序解耦的信用分配**：引入**Hindsight Relabeling**和**Counterfactual Credit Assignment**——对失败轨迹，回溯分析每一步的"反事实"（如果这一步做了不同选择，结果是否会改变？），为关键决策点提供丰富、低方差的训练信号。这与基于因果推断的信用分配（如Shapley值分解）方向一致，但在Agentic轨迹的规模下计算可行性仍是瓶颈。

3. **Online PRM训练**：在RL训练循环中同步训练或微调Process Reward Model，使其持续适应策略分布的变化，为每步工具调用提供密集过程奖励。这与On-Policy Distillation的思路一致——所有辅助模型都应与策略保持同步更新，避免分布偏移导致的信号失效。

4. **Multi-Agent RL视角**：将Agentic LLM的推理循环建模为多Agent协作——"规划者"、"执行者"、"验证者"各自是一个子策略，共享底层LLM但使用不同的system prompt或LoRA adapter，通过集中训练分散执行（CTDE）的方式联合优化。这种分解可以降低单策略的探索负担，并通过角色分工自然缓解信用分配问题——验证者负责评估执行者的输出质量，规划者负责根据验证结果调整计划。

## On-Policy Distillation
On-Policy Distillation 是将知识蒸馏技术引入 RL 训练循环的一种方法：在 RL 的 rollout 阶段，使用**当前策略**（而非固定的离线数据集）动态生成训练样本，再由教师模型对学生模型进行在线蒸馏。

### 与标准知识蒸馏的区别
标准 Knowledge Distillation（Hinton et al., 2015）使用预先收集的静态数据集，教师模型的软标签（soft label）一次生成、反复使用。这在 RL 场景下会引发**分布偏移（Distribution Shift）**问题——随着策略持续更新，模型的输出分布不断演化，而离线蒸馏的数据分布原地踏步，逐渐与当前策略脱节，导致蒸馏信号失效。

On-Policy Distillation 的核心思路是：每个 RL 训练步骤中，用当前策略采样新的轨迹，教师模型实时对这批数据打分并生成软标签，学生模型再在这批新鲜数据上更新参数。训练数据的分布始终与当前策略同步，从根本上消除分布偏移。

### 与 PPO/GRPO 的关系
On-Policy Distillation 可作为 RL 训练的**正则化手段**：在策略梯度损失之外，叠加教师模型与学生模型输出分布之间的 KL 散度作为辅助损失，防止策略在探索过程中崩溃（policy collapse）。这与 PPO 的 clip 操作和 KL 惩罚项思路一脉相承，但信号来源从参考模型变为了动态更新的教师模型。

### 代表性应用
**DeepSeek-R1** 的蒸馏流程是典型案例：先用 RL 训练出具有强推理能力的大模型（R1），再将其作为教师，通过 on-policy 的方式将推理轨迹蒸馏到 1.5B/7B 等小模型，小模型在 MATH 等基准上的表现接近甚至超越了同规模 SFT 训练的上限。此外，在 RLHF 中引入在线知识蒸馏（Online KD）也是当前的研究热点，旨在同时保留大模型的通用能力与 RL 带来的对齐增益。

### SFT ↔ OPD ↔ RL 三角关系

理解 On-Policy Distillation 的最佳方式，是将其置于 SFT 和 RL 构成的训练光谱中审视：

| 维度 | SFT | OPD | RL |
|------|-----|-----|-----|
| **数据来源** | 静态离线数据集 | 当前策略在线采样 | 当前策略在线采样 |
| **优化目标** | 交叉熵（模仿教师分布） | 策略梯度 + 蒸馏损失混合 | 纯策略梯度（奖励最大化） |
| **分布同步** | ✗ 训练数据固定不变 | ✓ 数据分布随策略同步更新 | ✓ 数据分布随策略同步更新 |
| **优化稳定性** | 极高（凸优化近似） | 高（教师信号提供正则化） | 低（奖励稀疏、方差大） |
| **探索能力** | 无（仅模仿） | 中等（RL 驱动探索，教师兜底） | 强（完全自主探索） |

- **SFT = 离线蒸馏**：本质上是在静态数据集上做监督学习，通过交叉熵损失最小化学生与教师输出分布的差异。SFT 相当于用教师的"标准答案"训练学生，稳定但完全受限于数据集质量，学生不会超出教师的能力边界。
- **RL = 奖励驱动的策略搜索**：模型自主采样、环境（或 Reward Model）打分、策略梯度更新。RL 能突破教师天花板、涌现新能力，但训练极不稳定——奖励信号的稀疏性、高方差以及策略的快速漂移常常导致崩溃。
- **OPD = 两者的平衡点**：保留了 RL 的在线采样机制（数据分布始终与当前策略对齐），同时继承了 SFT 的蒸馏稳定性（教师模型提供平滑、低方差的训练信号）。形象地说，**OPD 用 RL 的"发动机"保证探索动力，用蒸馏的"方向盘"保证不跑偏**。

### OPD 作为平滑的 SFT → RL 过渡

在 RL 训练的早期阶段，策略还远未收敛，此时直接进行策略梯度优化面临严峻挑战：

1. **奖励信号极度稀疏**：模型生成的轨迹质量参差不齐，大部分采样结果奖励为零或极低，只有极少数的"幸运"采样能获得有效奖励信号，导致梯度估计方差巨大。
2. **策略崩塌风险**：有限的奖励信号驱动模型朝某个局部方向快速收敛，极易造成输出分布坍缩——模型开始反复输出少数几个"高分"模式，丧失生成多样性。
3. **灾难性遗忘**：策略梯度优化可能覆盖 SFT 阶段学到的通用语言能力，出现"对齐税"（alignment tax）——对齐效果上去了，但基础能力下降了。

OPD 为此提供了一个**"安全腰带"机制**：在训练初期，教师模型的蒸馏损失占主导地位，确保学生即使探索失败也不会崩溃——教师信号始终为学生提供一个"安全基线"。随着训练的推进，逐步降低蒸馏损失的权重、提高策略梯度损失的权重（即**退火策略，annealing**），让学生在稳定过渡中逐渐学会自主探索。

典型的退火调度公式为：

$$
\alpha\_t = \alpha\_{\text{start}} \cdot \gamma^{t/T} + \alpha\_{\text{end}} \cdot (1 - \gamma^{t/T})
$$

其中 $\alpha\_t$ 为第 $t$ 步的蒸馏损失权重，从高初值 $\alpha\_{\text{start}}$（如 0.8~0.9，蒸馏主导）逐步衰减至终值 $\alpha\_{\text{end}}$（如 0.1~0.2，RL 主导），$\gamma$ 控制衰减速率，$T$ 为总训练步数。这种渐进式的"松绑"策略在实践中被证明能显著降低 RL 训练的失败率，尤其适合从 SFT 模型初始化起的端到端训练。

### KL 正则化的等价性分析

OPD 的蒸馏损失与 PPO 的 KL 惩罚项在数学形式上高度相似，但两者存在关键差异。

**PPO 的 KL 惩罚**：在奖励函数中减去当前策略 $\pi\_\theta$ 与参考策略 $\pi\_{\text{ref}}$ 之间的 KL 散度：

$$
r\_{\text{KL-penalized}} = r(s, a) - \beta \cdot D\_{\text{KL}}(\pi\_\theta(\cdot|s) \parallel \pi\_{\text{ref}}(\cdot|s))
$$

其中 $\pi\_{\text{ref}}$ 通常是训练开始前的 SFT 模型（或上一轮迭代的快照），**保持冻结不变**。其作用是防止策略偏离初始分布太远，本质上是一个"回归原点"的约束。

**OPD 的蒸馏损失**：直接在损失函数中加入学生策略 $\pi\_S$ 与教师策略 $\pi\_T$ 之间的 KL 散度：

$$
\mathcal{L}\_{\text{distill}} = D\_{\text{KL}}(\pi\_T(\cdot|s) \parallel \pi\_S(\cdot|s))
$$

这里的核心区别在于：**教师 $\pi\_T$ 是一个更强的、动态更新的模型**（如经过完整 RL 训练的 R1），其输出分布本身就是高质量的优化目标，而非仅仅是一个"不回退"的锚点。

两者对比：

| 特性 | PPO KL 惩罚 | OPD 蒸馏损失 |
|------|-----------|------------|
| **参考模型** | $\pi\_{\text{ref}}$（冻结旧策略） | $\pi\_T$（更强的教师模型） |
| **目标方向** | 约束：不要离起点太远 | 引导：朝更好的方向移动 |
| **模型能力** | 通常 ≤ 当前策略 | 通常 > 当前策略 |
| **更新方式** | 训练全程冻结 | 可静态也可迭代更新 |

**何时冗余、何时互补？** 如果教师模型 $\pi\_T$ 本身就远强于学生，蒸馏损失已经天然包含了"不退化"的约束（往教师方向靠拢本身就是一种正则化），此时再叠加 PPO 式的 ref-model KL 惩罚是**冗余的**，甚至可能因过度约束拖慢优化速度。但当教师与学生能力接近（如同规模的 online distillation），蒸馏信号的引导力不足时，额外的 ref-model KL 惩罚作为**互补正则化**有助于防止策略在低置信度区域剧烈震荡。

### OPD 损失函数构成

On-Policy Distillation 的完整损失函数由策略梯度损失和蒸馏损失加权组合而成：

```python
def opd\_loss(
    student\_logits,       # 学生模型输出的 logits
    teacher\_logits,       # 教师模型输出的 logits（同 batch 在线生成）
    old\_log\_probs,        # 采样时学生模型的 log prob（用于重要性采样）
    advantages,           # 优势函数（GAE 或 GRPO 组内标准化）
    actions,              # 实际采样的 token 序列
    alpha=0.5,            # 蒸馏损失权重（可随训练步数退火）
    clip\_eps=0.2,         # PPO clip 阈值
    kl\_beta=0.1,          # （可选）额外 ref-model KL 惩罚系数
):
    # ---- 1. 策略梯度损失（PPO-clipped） ----
    new\_log\_probs = F.log\_softmax(student\_logits, dim=-1)
    action\_log\_probs = new\_log\_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    ratio = (action\_log\_probs - old\_log\_probs).exp()  # 重要性采样比率
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip\_eps, 1 + clip\_eps) * advantages
    policy\_loss = -torch.min(surr1, surr2).mean()

    # ---- 2. 蒸馏损失（教师→学生 KL 散度） ----
    student\_log\_probs = F.log\_softmax(student\_logits / temperature, dim=-1)
    teacher\_probs = F.softmax(teacher\_logits / temperature, dim=-1)
    distill\_loss = F.kl\_div(
        student\_log\_probs, teacher\_probs, reduction="batchmean"
    )

    # ---- 3. （可选）额外 ref-model KL 惩罚 ----
    if ref\_logits is not None:
        ref\_probs = F.softmax(ref\_logits / temperature, dim=-1)
        ref\_kl = F.kl\_div(student\_log\_probs, ref\_probs, reduction="batchmean")
    else:
        ref\_kl = 0.0

    # ---- 4. 组合损失 ----
    # alpha 采用退火调度：从 ~0.9（蒸馏主导）逐步衰减至 ~0.1（RL 主导）
    total\_loss = (1 - alpha) * policy\_loss + alpha * distill\_loss + kl\_beta * ref\_kl

    return total\_loss
```

关键设计要点：
- **$\alpha$ 退火**：`alpha` 不是固定值，而是随训练步数按指数衰减。早期 $\alpha$ 大，模型主要模仿教师，RL 信号仅作微调；后期 $\alpha$ 小，模型开始自主探索，教师仅提供兜底正则化。
- **temperature 共享**：蒸馏部分的 softmax 温度通常与教师推理时一致，保证分布平滑度匹配。
- **ref\_kl 可选**：当教师远强于学生时，`ref\_kl` 通常置零以免冗余约束；当教师与学生能力接近时，保留 `ref\_kl` 作为额外稳定项。

## 数据工程（Data Curation）

预训练数据的质量决定了 LLM 能力的**天花板**——模型架构和训练算法只是逼近这一天花板的手段。数据工程的核心理念可以浓缩为一句话：**15T 高质量 token 胜过 50T 未过滤 token**（FineWeb, Penedo et al., 2024）。这一发现深刻重塑了 Scaling Laws 的实践含义：单纯堆数据量而不控制质量，不仅是效率浪费，更可能引入噪声和偏见，损害模型在下游任务上的表现。

### 1. 数据质量 > 数据数量：实证证据

FineWeb 的消融实验揭示了数据质量的决定性作用：

| 实验设置 | 数据量 | 过滤策略 | 下游困惑度 (↓) | MMLU (↑) |
|----------|:------:|----------|:------------:|:--------:|
| 原始 Common Crawl | 50T tokens | 无 | 9.2 | 48.3% |
| C4 级过滤 | 30T tokens | 基本启发式 | 8.7 | 52.1% |
| FineWeb 全流程 | 15T tokens | 启发式 + 模型打分 + 去重 | **8.1** | **57.8%** |

15T 高质量 token 在所有指标上均显著优于 50T 原始数据，且训练时间缩短 70%（更少的 token 意味着更少的训练步数）——**质量即效率**。

DCLM（DataComp for Language Models, Li et al., 2024）进一步系统化地证明了：在固定训练计算预算下，数据过滤策略的选择与模型架构设计**同等重要**——最优过滤策略可以将同架构模型的 benchmark 分数提升 5-8 个百分点，这一增益量级堪比将模型参数量翻倍。

### 2. 质量过滤流水线：四层漏斗

现代大规模数据过滤采用多层流水线，每层逐步剔除不同类别的低质量数据：

| 过滤层 | 方法 | 剔除目标 | 通过率 |
|--------|------|----------|:------:|
| **(a) 启发式过滤** | 困惑度阈值、URL 黑名单、语言检测（fastText）、长度截断（<100 或 >100K tokens 丢弃）、特殊字符比例 | 明显垃圾、非目标语言、过短/过长文档 | ~60-70% |
| **(b) 模型质量打分** | 在高质量语料（Wikipedia/Books）上训练 fastText 或轻量 KL 分类器，对每条文档打分 | 格式混乱、语义不连贯、低信息密度 | ~50-80%（叠加后） |
| **(c) 精确去重** | 文档级 SHA256/SimHash → 段落级精确字符串匹配 | 重复数据（降低多样性、引入 memorization 风险） | ~85-95% |
| **(d) 近似去重 & 多样性** | MinHash LSH（$k=256$ 哈希函数，$b=16$ bands）检测高重合度文档对（Jaccard ≥ 0.8 → 删一留一）；语义去重（embeddings + 聚类） | 模板化内容、SEO 垃圾、语义冗余 | ~70-90% |

**MinHash LSH 的数学原理**：对于两个文档的 token n-gram 集合 $A, B$，MinHash 保持 Jaccard 相似度的无偏估计：

$$\hat{J}(A, B) = \frac{1}{k}\sum\_{i=1}^{k} \mathbb{1}[\min\_{a \in A} h\_i(a) = \min\_{b \in B} h\_i(b)]$$

$k$ 越大估计越精确；LSH（Locality Sensitive Hashing）通过 $b$ 个 band 将 $k$ 个哈希值分桶，仅对同一桶内的文档对计算完整 Jaccard，将复杂度从 $O(n^2)$ 降至 $O(n^{1+1/b})$——这对 TB 级语料去重至关重要。

### 3. 合成数据流水线：从蒸馏到自我演化

当自然数据质量达到瓶颈时，合成数据成为突破天花板的关键手段。三种主要范式：

| 范式 | 流程 | 优势 | 局限 |
|------|------|------|------|
| **(a) 强模型蒸馏** | GPT-4/Claude 生成 → 质量过滤 → SFT 训练小模型 | 质量上限高，适合 SFT 数据 | 受限于教师能力，无法超越 |
| **(b) Self-Play 生成** | 当前模型生成 → 自过滤（奖励模型打分）→ 保留高分样本 → 重新训练 | 可迭代自我提升 | 需要可靠的过滤机制防止退化 |
| **(c) Constitutional AI** | 模型生成 → 按宪法原则自我 critique → 迭代修订 → 保留修订后版本 | 内化对齐原则，无需人工标注 | 宪法设计本身就是工程挑战 |

**与 On-Policy Distillation（OPD）的关系**：合成数据流水线本质上是**离线（offline）**版本的数据增强——数据生成和训练是分离的两个阶段。OPD 则是**在线（online）**版本——在训练过程中实时从教师策略采样，数据分布随教师策略同步更新。二者的关键区别在于分布偏移（distribution shift）的处理：离线合成数据一旦生成即固定，可能随学生模型更新而出现 teacher-student 分布不一致；OPD 通过在线采样持续消除这一 gap。

数学上，离线合成数据的损失为：

$$\mathcal{L}\_{\text{offline}} = \mathbb{E}\_{(x,y) \sim \mathcal{D}\_{\text{teacher}}}[\log \pi\_\theta(y|x)]$$

而 OPD 的损失为：

$$\mathcal{L}\_{\text{OPD}} = \mathbb{E}\_{x, y \sim \pi\_T(\cdot|x)}[\log \pi\_\theta(y|x)]$$

区别看似微小（$\mathcal{D}\_{\text{teacher}}$ 是固定数据集 vs $\pi\_T$ 是动态策略），但在大规模训练中，$\pi\_T$ 可以采用与学生策略更新频率同步的**滚动采样**，使得 $y$ 的分布始终与当前训练的 $\pi\_\theta$ 保持最优 KL 距离。

### 4. 数据配比与课程学习

**领域权重配比**是预训练数据工程中最具争议和艺术性的环节。当前社区中的经验法则：

| 领域 | 典型比例 | 理由 |
|------|:------:|------|
| Web (通用文本) | 50-60% | 覆盖广泛知识面，语言表达能力基础 |
| Code | 15-20% | 提升推理和结构化思维，transfers to non-code tasks |
| Math | 5-10% | 强化符号推理，与编程产生协同效应 |
| 书籍 / 学术 | 10-15% | 提供高质量长文本，改善长程依赖 |
| 多语言 | 5-10% | 确保非英语能力 |

但**最优比例高度依赖于模型规模、架构和下游目标**——不存在普适「黄金比例」。Meta 的 LLaMA 3 在 ~1:1:3 的 code:math:web 附近进行了大量消融实验，但最终配比是通过在多个下游 benchmark 上的网格搜索确定的。

**课程学习（Curriculum Learning）**在预训练中的应用：不同于传统的一次性混合所有数据，课程学习按照难度递增的顺序逐步引入训练数据：

$$\mathcal{D}^{(t)} = \underbrace{\mathcal{D}\_{\text{easy}}}\_{\text{阶段1: 干净/高质}} \xrightarrow{\text{逐步混入}} \underbrace{\mathcal{D}\_{\text{medium}}}\_{\text{阶段2: 混合}} \xrightarrow{\text{逐步混入}} \underbrace{\mathcal{D}\_{\text{hard/dirty}}}\_{\text{阶段3: 全分布}}$$

实际效果：早期阶段用 Wikipedia + Books 等高质量语料奠定语言基础 → 中期混入经过精细过滤的 Web 数据 → 后期加入含噪声的原始分布以提升鲁棒性。课程学习在 Chinchilla 级别训练数据量下的提升约为 1-3% benchmark 分数——不算颠覆性，但足够显著。

**SFT 数据：质量的统治地位**。与预训练数据不同，SFT 数据的质量远重于数量——这一规律已被 LIMA（Zhou et al., 2023）和多个后续工作反复验证：**1K 条精心手写的 SFT 示例，在下游表现上一致优于 100K 条中等质量的示例**。原因在于 SFT 只需要少量高质量信号即可完成「从续写到对话」的行为转换，而低质量示例反而引入噪声和错误的行为模式。这一观察直接支撑了「从 SFT 到 RL」的统一框架——SFT 阶段仅需少量高质量数据做定位（行为克隆），真正的能力提升靠 RL 阶段的奖励驱动探索。

### 5. 数据污染：Benchmark 泄露的幽灵

数据污染（Data Contamination）是 LLM 评测中最棘手的开放问题——当预训练数据意外包含了下游 benchmark 的测试样本时，评测分数就失去了衡量泛化能力的意义。

**核心机制——Canary String 问题**：即使训练时使用了去重和过滤，benchmark 问题仍可能以改写、翻译、嵌在讨论中等形式出现在训练语料中。经典检测方法包括：

- **N-gram 重叠检测**（$n=8$ 或 $n=13$）：将 benchmark 样本切分为 n-gram，在训练语料中检索命中率。阈值通常设为 80% 重叠即判定为污染。
- **困惑度异常检测**：受污染样本的模型困惑度异常低（模型「背诵」了答案），通过统计 $z$-score 偏离度检测。
- **String Subset Matching**：去除标点和停用词后做最长公共子序列（LCS）匹配。

**污染的后果**：被污染的 benchmark 会虚增所谓的「涌现能力」——看似模型在某一规模突然获得某项能力，实际可能是恰好记住了训练数据中的对应样本。这直接扭曲了 Scaling Laws 的实践指导：如果 benchmark 被污染，$L(N, D)$ 曲线在污染点会出现虚假的「相变」，导致研究者将资源错误分配到看似有前景实际是伪信号的 scale 上。

### 6. 与本文其他章节的连接

| 连接点 | 关联内容 |
|--------|----------|
| **SFT & Alignment** | 数据质量直接决定 SFT 的行为模仿精度和 Alignment 的奖励信号可靠性——垃圾数据训练出的 reward model 必然给出带偏见的奖励，导致 RL 优化朝错误方向收敛 |
| **Scaling Laws** | FineWeb 级别的数据质量是 Scaling Laws 成立的前提——Kaplan/Chinchilla 的 $L(N, D)$ 公式假设数据质量恒定，若数据质量随 $D$ 增大而下降，真实损失曲线将偏离理论预测 |
| **Agentic RL** | 智能体与环境交互产生的工具输出、搜索片段、代码执行结果，本质上也是需要「数据策展」的新型语料——工具返回的噪声、截断、无关输出需要过滤和控制，否则污染策略梯度信号 |
| **Harness Engineering** | TB 级数据去重需要大规模流式处理管线（Apache Spark / Ray），去重算法（MinHash LSH）的计算复杂度约束和内存占用是工程实现的硬瓶颈 |

数据工程的本质可以这样理解：如果说 Scaling Laws 定义了「给定数据和模型，loss 能降到多低」，那么数据策展定义的就是「给定计算预算，我们实际接触到的数据离最优分布有多近」——前者是理论极限，后者是工程逼近。


## Harness Engineering
Harness Engineering 指支撑大规模 LLM RL 训练的工程基础设施体系。RL 训练不同于监督学习的离线批处理，需要推理（rollout）与训练（training）的紧密配合，对系统吞吐量、延迟和资源调度的要求极高。

### 核心组件

**Rollout 引擎**是 RL 训练的数据来源。在每个训练迭代中，需要用当前策略对大量 prompt 并发采样，生成带奖励标注的轨迹。主流方案基于 **vLLM** 或 **SGLang** 等高性能推理框架，利用 PagedAttention 和连续批处理技术将 GPU 利用率提升至接近理论上限。

**Reward Model 服务**负责对 rollout 轨迹进行实时打分。规则奖励（如答案正确性验证、格式检查）可在 CPU 上高效完成；神经网络奖励（RM/PRM）则需要独立的推理服务以降低延迟。两类奖励通常并行计算后加权融合。

**训练引擎**承载 Actor/Critic 模型的参数更新，通常基于 **DeepSpeed**（ZeRO-3 offloading）或 **Megatron-LM**（张量并行 + 流水线并行）实现分布式训练，以容纳百亿级别参数的梯度计算。

**数据流水线**管理 rollout 结果的收集、过滤（去除截断样本、奖励方差为零的组）和 replay buffer 的维护，需要在推理节点和训练节点之间进行高效的跨节点数据传输。

**实验管理**包括 checkpoint 的版本控制、超参数追踪（WandB / MLflow）以及 A/B 实验的对照管理，是保障大规模实验可复现性的重要基础设施。

### 工程挑战
RL 训练中 rollout 和 training 对硬件的需求截然不同：rollout 追求高吞吐、低显存占用（适合 A100/H100 with FP8），training 追求高带宽互联和大批量梯度聚合（适合 H100/B200 NVLink）。异构硬件的资源调度与负载均衡是核心难点。显存优化方面，gradient checkpointing、ZeRO offloading 和 activation recomputation 是标配手段，但会引入额外的计算开销，需要精细调参以找到最优的吞吐 - 显存平衡点。

### 代表性框架
- **OpenRLHF**：基于 Ray + vLLM + DeepSpeed 的开源 RLHF 框架，支持 PPO/GRPO 等多种算法，架构清晰，适合学术研究。
- **verl**（ByteDance）：字节跳动开源的高性能 RL 训练框架，强调 rollout 与 training 的解耦调度，生产环境验证充分。
- **trl**（HuggingFace）：HuggingFace 生态的 RLHF 工具库，与 transformers 深度集成，上手门槛低，适合快速原型验证。
- **DeepSpeed-Chat**：微软提供的端到端 RLHF 解决方案，原生支持 ZeRO 优化和混合精度训练。
