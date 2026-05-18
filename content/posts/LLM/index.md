---
date: '2026-04-19T16:36:00+08:00'
draft: false
title: 'LLM基础知识'
---
*由于最近在准备LLM相关的实习，但是在面试过程中发现对于一些理论还是不太熟悉，导致经常卡壳，所在就在这里记录一下这段时间学到的关于LLM的知识，也是帮助自己巩固一下。*

## LLM的本质
LLM, Large language model本质上就是集于Transformer在做自回归的任务（即预测下一个token），LLM的训练可以分为三个部分：
- **预训练（Pretraining）**：预训练的过程就是让模型学会如何生成语言，所以本质上预训练模型的任务目标是续写，在具体的训练过程中，输入数据为大量的语言数据，模型通过自监督的训练方式预测下一个token的概率分布，然后通过交叉熵损失来优化整个模型。
- **监督微调（SFT，Supervised FineTuning）**：预训练的模型已经拥有了生成的能力，但是缺乏对话的能力，而SFT的过程就是让模型学会对话。SFT和Pretraining的训练过程以及损失函数是一致的，不同之处在于，SFT的训练数据是对话形式的，我们不需要去学习生成问题的能力，而是要根据问题来生成答案，因此对问题（Prompt）去做交叉熵损失就是没有必要的，SFT的做法是做一个mask，将Prompt部分的token id赋值为-100（交叉熵损失会将其忽略）。并且由于在Pretraining的过程中，我们已经得到一个非常强大的语言生成模型，我们在SFT的过程中就不太需要训练很多轮次。
- **对齐（Alignment）**：在SFT中模型学会了对话的能力，但是对话生成的答案质量往往取决于SFT训练数据集的质量，如果遇到训练数据外的分布，那么就会出现严重的幻觉。因此，我们不仅需要让LLM学会对话，还要让LLM学会如何生成一个好的答案，这就是LLM对齐要做的事。

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

$$\pi_\theta(y|x) = \prod_{t=1}^{T} \pi_\theta(y_t|x, y_{<t})$$

**Direct Response**（无CoT）直接将问题映射到答案，搜索空间仅为输出序列的所有可能组合。对于复杂推理问题（如数学证明、多步逻辑），正确路径在直接映射下是概率极低的"针尖"，模型极易在错误的捷径上收敛。

**CoT**插入推理token序列 $\{r_1, r_2, ..., r_K\}$ 作为中间状态，将生成过程分解为：

$$\pi_\theta(y|x) = \sum_{r_1, ..., r_K} \pi_\theta(r_1|x) \cdot \pi_\theta(r_2|x, r_1) \cdots \pi_\theta(y|x, r_1, ..., r_K)$$

这些中间推理步骤起到两个关键作用：(1) **状态分解**——将"问题→答案"的长程跳跃分解为"问题→中间结论→中间结论→答案"的短程级联，每一步的条件概率显著更高；(2) **自修正路径**——中间步骤允许模型输出不确定性（"这可能是错误的，让我重新检查..."），从而在生成过程中动态调整后续推理方向。这本质上是在条件概率图上开启了更多可达路径，将搜索从宽度优先的隐式空间展开为深度优先的显式轨迹。

### System 1 vs System 2：推理的工程实现

Kahneman的"双系统"认知理论在LLM推理中有清晰的工程映射：

| 维度 | System 1（快速直觉） | System 2（慢速推理） |
|------|---------------------|---------------------|
| **输出模式** | 直接生成答案，无中间推理 | 逐步推理 + 最终答案 |
| **推理时机制** | 单次前向传播（1 pass） | 多次自回归解码 + 验证 |
| **token效率** | 高（仅输出答案token） | 低（推理token数 ≫ 答案token数） |
| **准确率** | 低（复杂问题失效） | 高（可处理多步推导） |
| **计算成本** | $O(T_{\text{ans}})$ | $O(T_{\text{ans}} + T_{\text{reason}})$ |
| **对应训练方法** | 标准SFT | SFT + RL（CoT数据 + 推理奖励） |
| **代表模型** | GPT-4 (direct), Claude (无thinking) | o1, o3, DeepSeek-R1, Claude Extended Thinking |

**工程实现的关键决策**是何时触发System 2。当前主流方案包括：(1) **路由分类器**——用小模型判断问题难度，简单问题走快速通道（System 1），复杂问题走推理通道（System 2），如 DeepSeek-V3 的 MoE 路由思想在推理层面的延伸；(2) **推理预算token**——强制分配固定数量的thinking token（如Claude Extended Thinking的budget参数），模型在预算内自由组织推理过程；(3) **自适应终止**——模型自主判断推理是否充分（通过输出结束标记或置信度阈值），无需人工预设推理长度。

更前沿的方向是**推理-行动交织（Interleaved Reasoning-Action）**：System 2不只是"先想后答"，而是在推理过程中动态穿插工具调用——推理到某一步时调用计算器验证数值，或调用代码执行器运行片段代码——将语言推理与外部验证实时融合。ReAct和Tool-Integrated CoT是该范式的早期实现。

### 推理长度的经济学：成本与精度的权衡

推理长度是LLM推理中最核心的**可控经济变量**。长推理链虽然提升准确率，但带来三重成本：

**1. 延迟成本**：$T_{\text{reason}}$ 个推理token的生成时间是纯答案生成的 $\frac{T_{\text{reason}}}{T_{\text{ans}}}$ 倍。对于需要实时交互的场景，每条查询多等2秒即可显著降低用户体验。

**2. 显存/吞吐成本**：长推理链的KV cache与序列长度成正比 $O(L \cdot d_{\text{model}} \cdot n_{\text{layers}})$。在大规模在线服务中，一条10K token的推理请求可能占用等同于5条2K token请求的显存，直接削减服务吞吐量。

**3. 金钱成本**：API调用按token计费，推理token通常与输出token同价。一条包含2000-token推理链的回答可能比直接回答贵5-10倍。

**长度惩罚的数学形式**：设推理链长度为 $L$，最终答案正确性得分为 $R_{\text{acc}}$，则总奖励可建模为：

$$R_{\text{total}}(L) = R_{\text{acc}}(L) - \lambda \cdot L$$

其中 $R_{\text{acc}}(L)$ 通常是 $L$ 的凹函数——推理长度带来的精度提升存在边际递减效应（前200个推理token可能大幅提升准确率，之后每增加100个token的边际收益迅速衰减）。$\lambda$ 是长度惩罚系数，控制 cost-accuracy trade-off 的斜率。

**最优推理长度的求解**：在给定问题难度下，最优推理长度 $L^*$ 应满足一阶条件：

$$\frac{\partial R_{\text{acc}}}{\partial L}\bigg|_{L=L^*} = \lambda$$

即当推理的边际精度收益恰好等于边际成本时，总效用最大。实践中 $R_{\text{acc}}(L)$ 的具体函数形式取决于模型能力和任务难度——对简单问题，$R_{\text{acc}}(L)$ 在很小的 $L$ 处就接近饱和；对复杂问题，$R_{\text{acc}}(L)$ 可能持续增长数百至数千个token。

这一公式也解释了**动态推理预算**的动机：当模型能准确估计问题难度时，可自适应选择 $L^*$——简单问题 $L^* \approx 0$（跳过推理，即System 1），中等难度 $L^* = 200\sim500$，高难度 $L^* = 2000+$。

Efficient Reasoning方法：长推理链虽然准确，但对简单问题会产生大量冗余token，增加推理延迟和计算成本。高效推理的目标是在保证质量的前提下降低推理开销，主要方法包括：
1. **动态推理预算（Thinking Budget）**：根据问题难度动态调整推理链长度，对简单问题缩短推理过程，对复杂问题保持完整推理。数学上等价于对每个查询求解 $\arg\max_L [R_{\text{acc}}(L|x) - \lambda L]$，其中 $R_{\text{acc}}(L|x)$ 为条件于问题 $x$ 的精度函数。Claude 3.7 引入了 Extended Thinking 的预算控制机制，可在准确性与延迟之间灵活权衡。
2. **推理蒸馏（Reasoning Distillation）**：通过SFT将长推理链压缩为更短但等效的推理路径，剔除冗余的重复验证步骤，常用于将大模型推理能力迁移到小模型。本质是以教师模型的 $R_{\text{acc}}(L)$ 曲线为监督信号，训练学生模型以更小的 $L$ 达到同等的 $R_{\text{acc}}$。
3. **Speculative Decoding**：利用小模型（draft model）快速生成候选token序列，再由大模型并行批量验证，在不降低输出质量的前提下显著提升生成速度。推理token的验证可按 $k$ 个token并行，将 $O(L)$ 的串行延迟降低为 $O(L/k)$。
4. **Latent Reasoning（COCONUT, Chain of Continuous Thought）**：将推理过程从离散的语言空间迁移到连续的隐空间（latent space），用最后一层隐状态直接作为下一步的"思维"输入，而非解码为token，大幅减少推理所需的token数量，在某些多步推理任务上性能不降反升。这从根本上绕过了token级的自回归瓶颈，是在推理效率上的范式级突破。


## Policy Optimization

### 从SFT到RL：优化视角的统一

SFT与RL并非两种截然不同的范式，从优化目标出发，二者可以统一在同一个数学框架下理解。

**SFT的本质是行为克隆（Behavior Cloning, BC）**：给定专家数据集 $\mathcal{D} = \{(x, y^*)\}$，SFT通过最大化条件对数似然 $\max_\theta \mathbb{E}_{(x,y^*)\sim\mathcal{D}}[\log \pi_\theta(y^*|x)]$ 来训练策略。这等价于在专家状态分布上进行监督学习——模型只见过专家的"正确路径"，从未被要求评估或比较其他候选回复。**SFT的优势在于训练稳定**（目标函数为凸优化问题，梯度方向明确），但**致命缺陷是分布外（OOD）崩塌**：当推理时遇到训练分布未覆盖的输入模式，模型的行为完全不可控，容易产生幻觉或低质量回复。

**RL的做法是奖励驱动的最优策略搜索**：引入奖励函数 $r(x,y)$（来自Reward Model或规则验证器），优化目标变为 $\max_\theta \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(\cdot|x)}[r(x,y)]$。RL允许模型**主动探索**——对同一输入采样多条不同的输出，通过奖励信号区分优劣，再用策略梯度（Policy Gradient）的方式将高奖励输出的概率推高。**重要性采样（Importance Sampling）** 是实现这一目标的核心数学工具：

$$\max_\theta \mathbb{E}_{x,y\sim\pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(y|x)}{\pi_{\theta_{\text{old}}}(y|x)} \cdot A(x,y) \right]$$

当重要性比率 $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} \equiv 1$（即不做任何策略更新，仅在当前策略分布上做加权），RL损失退化为带优势权重的SFT损失。**SFT是RL在"固定分布、无探索"条件下的退化特例**——SFT只在专家数据上拟合（优势恒为1，比率恒为1），而RL在模型自主生成的分布上（比率≠1）利用优势信号区分好坏，打破行为克隆的分布束缚。

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

**优势函数** $A_t$ 的构建是PPO的核心。PPO以Critic模型估计的状态价值 $V(s_t)$ 为baseline，通过**GAE（广义优势估计）** 计算未来每一步的TD误差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，然后进行指数衰减的加权求和：

$$A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

其中 $\gamma$ 控制远期奖励的衰减，$\lambda$ 在偏差（$\lambda=0$，仅用单步TD误差）和方差（$\lambda=1$，用完整蒙特卡洛回报）之间做 bias-variance trade-off。此外，每一步的奖励中会加入与Ref模型的**KL散度惩罚项** $-\beta \cdot \text{KL}(\pi_\theta \parallel \pi_{\text{ref}})$，防止Actor在探索过程中偏离初始策略太远。

PPO的关键操作为**clip机制**。记重要性比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$，PPO的目标函数为：

$$\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

clip将比率强行限制在 $[1-\epsilon, 1+\epsilon]$ 内（典型 $\epsilon=0.2$），$\min$ 操作保证当 $A_t>0$ 时不因过度增大比率而获得虚假收益，当 $A_t<0$ 时不因过度减小比率而逃避惩罚。这从根本上**防止策略在单步更新中变化过猛导致策略崩溃（policy collapse）**——一旦新旧策略差异过大，clip直接截断梯度，模型在该样本上停止更新。
### DPO（Direct Policy Optimization）
DPO的核心思想是绕过传统的reward model和复杂的PPO强化学习阶段，直接使用人类偏好数据对语言模型进行优化。

它通过损失函数拉大模型生成chosen回复和rejected回复的概率差，避免了PPO训练时的在线采样成本以及Actor-Critic架构的崩溃风险。

```python
def dpo_loss(chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta=0.1):
    chosen_logratio = chosen_logp-chosen_ref_logp
    rejected_logratio = rejected_logp-rejected_ref_logp
    logratio = chosen_logratio-rejected_logratio
    loss = -F.logsigmoid(beta*logratio)
    return loss.mean()
```

### GRPO（Group Relative Policy Optimization）

#### 统一的变体分析框架

PPO、GRPO、DAPO、GSPO等算法并非彼此孤立，而是基于**同一套数学积木**在不同维度上的差异化组合。统一框架的核心组件为：

$$\mathcal{L} = \mathbb{E} \left[ \min\left( \underbrace{r(\theta)}_{\text{重要性比率}} \cdot \underbrace{A_i}_{\text{优势}},\ \underbrace{\text{clip}(r(\theta))}_{\text{截断比率}} \cdot A_i \right) - \beta \cdot \underbrace{D_{\text{KL}}}_{\text{KL惩罚}} \right]$$

所有变体共享这三大积木，区别仅在三个维度：(1) **优势计算**——用Critic的GAE估计还是组内标准化；(2) **损失聚合粒度**——token级、序列级还是batch级求平均；(3) **clip参数**——上下界对称还是非对称，clip加在哪一层。下表总结了核心差异：

| 维度 | PPO | GRPO | DAPO | GSPO |
|------|-----|------|------|------|
| 优势 $A_i$ | Critic + GAE | 组内Z-score标准化 | 组内Z-score标准化 | 组内Z-score + 累积奖励 |
| 损失聚合 | Token级平均 | Token级平均 | Token级直接聚合 | **序列级聚合** |
| Clip 参数 | $\epsilon=0.2$ 对称 | $\epsilon=0.2$ 对称 | **上界放大** (higher) | $\epsilon$ 对称，序列级 |
| Critic | ✅ 需要 | ❌ 不需要 | ❌ 不需要 | ❌ 不需要 |

#### 为什么GRPO取代PPO：Critic在LLM中的三大困境

GRPO的核心创新是**废除Critic模型**，直接用组内相对排名计算优势。这一设计的背后是Critic在LLM场景中的三个致命问题：

**1. 高方差（High Variance）**：LLM的生成空间极为庞大（每步从数万词表中采样），状态价值函数 $V(s)$ 的估计天然具有高方差。PPO的GAE是MC和TD的折中方案，但在长序列（数百至数千token）中，$\lambda$ 的 bias-variance trade-off 极难调参——$\lambda$ 偏高则方差爆炸，$\lambda$ 偏低则偏差让训练信号失去区分度。

**2. 额外容量开销（Extra Capacity）**：一个与Actor（动辄几十B参数）同规模的价值网络意味着近乎翻倍的显存和计算开销。在LLM RL训练中，Actor本身已占绝大部分GPU资源，再加一个等大小的Critic严重限制了batch size和序列长度。

**3. 参数共享冲突（Parameter Sharing Conflict）**：实践中常让Actor和Critic共享底层transformer参数以节省显存（仅顶层head不同）。但生成任务（next-token prediction）和价值估计（scalar regression）的优化方向相互矛盾——生成需要保留丰富语义细节，价值估计需要将语义压缩为单一标量。这种冲突会同时损害生成质量和价值估计的准确性。

GRPO用一个巧妙的统计替代方案解决了这三个问题：**对同一prompt采样一组（Group）$G$ 条回复，用组内均值与标准差做Z-score标准化获得相对优势**：

$$A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \varepsilon}$$

其中 $\mathbf{r} = \{r_1, r_2, ..., r_G\}$ 为组内$G$条回复的奖励得分。这一做法之所以有效，是因为**组内相对排序提供了一种无需估计绝对价值的"胜者通吃"信号**——在给定prompt下，模型只需知道哪些回复比其他更好，无需知道"多好"。统计学上，Z-score标准化退化为大数定律下的秩检验，当组大小$G$足够大时（通常$G=4\sim16$），组内标准化后的相对优势在期望上等价于真实优势的单调变换。

GRPO的损失函数为：

```python
def grpo_loss(rewards, logp_per_token, ref_logp_per_token, old_logp_per_token, mask=None, beta=0.01, clip_eps=0.2):
    mean_rewards = rewards.mean(dim=-1, keepdim=True)
    std_rewards = rewards.std(dim=-1, keepdim=True)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)
    advantages_per_token = advantages.unsqueeze(-1)

    ratio = torch.exp(logp_per_token-old_logp_per_token)
    adv1 = ratio*advantages_per_token
    adv2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)*advantages_per_token

    kl_per_token = torch.exp(ref_logp_token-logp_per_token)-(ref_logp_token-logp_per_token)-1

    loss_per_token = -(torch.min(adv1, adv2) - beta*kl_per_token)
    policy_loss = (loss_per_token*mask).sum(dim=-1)/mask.sum(dim=-1)
    return policy_loss.mean()
```

其中KL散度采用 **k3 估计器** $\text{KL} = \exp(z_\theta - z_{\text{ref}}) - (z_\theta - z_{\text{ref}}) - 1$（其中 $z = \log\pi$），这是KL散度的一个二阶泰勒展开近似形式，在 $z_\theta \approx z_{\text{ref}}$ 时精确成立。相比直接计算 $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$，k3估计器在数值上更稳定，且天然非负（$f(x)=\exp x - x - 1 \geq 0$ 当取等时 $x=0$）。

### DAPO（Decoupled clip and Dynamic sAmpling Policy Optimization）

DAPO在GRPO的基础上做出四项关键改进，每项改进都有明确的理论矛头所向：

**1. Higher Clip（上界放宽）**：GRPO采用对称clip $[1-\epsilon, 1+\epsilon]$，这对于正优势（$A_i>0$）和负优势（$A_i<0$）施加了对称的更新幅度限制。但问题在于：正优势样本是模型需要**强化**的高质量输出，负优势样本是需要**抑制**的低质量输出。对称clip下，一个奖励极高的样本（比率>>1+ε）会被截断——这意味着模型失去了最大化利用好样本的机会，策略多样性被不必要地压缩。DAPO将上界放大（如 $\epsilon_{\text{upper}} > \epsilon_{\text{lower}}$），允许模型对高奖励输出做更大步的强化更新，同时保持下界限制以防止过度惩罚。数学上体现为非对称clip：

$$\text{clip}(r(\theta)) = \min\left(r(\theta),\ 1+\epsilon_{\text{upper}}\right) \quad \text{而非} \quad \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)$$

**2. Dynamic Sampling（动态采样）**：GRPO的组内标准化依赖样本方差提供区分度——若组内所有回复的奖励相同（$\text{std}(\mathbf{r}) \approx 0$），则 Z-score 分母趋近于零，优势信号失效。这种情况在遇到极简单（所有回复全对）或极困难（全错）的prompt时频繁出现，称为**零方差问题（Zero-Variance Problem）**。DAPO的动态采样强制要求每组样本的奖励方差超过最低阈值：在采样阶段持续生成候选回复，过滤掉奖励完全一致的组，或在组内通过拒绝采样补充差异化样本，确保 $\text{std}(\mathbf{r}) > \tau_{\text{min}}$。理论上这等价于为策略梯度注入了有信息量的比较信号的最小充分条件。

**3. Token-Level Policy Gradient Loss（Token级直接聚合）**：这是四项改进中**理论动机最深**的一项。GRPO在序列级别计算平均loss：

$$\mathcal{L}_{\text{seq}} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{T_n} \sum_{t=1}^{T_n} \ell_{n,t}$$

其中 $N$ 为组内样本数，$T_n$ 为第 $n$ 个样本的token数。这种做法的隐蔽缺陷称为**长序列梯度稀释（Long-Sequence Gradient Dilution）**：对于一条高质量但很长的推理链（如包含反思和自我纠错的2000-token输出），每个token对梯度的平均贡献仅为 $1/2000$；而一条短回复（如直接给出答案的50-token输出）的token级贡献权重高达 $1/50$。这意味着**长推理链中的关键逻辑步骤会在反向传播中被系统性地低估**，模型难以学到高阶推理模式。

DAPO将聚合方式改为Token级直接求和后再做全局归一化：

$$\mathcal{L}_{\text{DAPO}} = \frac{\sum_{n,t} \ell_{n,t} \cdot \mathbb{I}[\text{mask}_{n,t}]}{\sum_{n,t} \mathbb{I}[\text{mask}_{n,t}]}$$

这确保每个有效token对梯度的贡献完全均等，消除了序列长度对学习信号强度的扭曲。对推理训练尤为关键——反思token（"Wait, this is wrong..."）和关键推导步骤获得了与其信息量匹配的梯度权重。

**4. Overlong Reward Shaping（超长惩罚塑形）**：RL训练中，模型会探索出"通过输出更长回复来刷分"的reward hacking行为——验证器可能对更长的回复给出虚假高分（例如包含更多正确片段，即使整体逻辑混乱）。DAPO的做法是：(a) **硬截断**：在损失计算中直接剔除被截断的样本（在生成阶段设置max_new_tokens，超出即truncate并在loss中mask掉全部token），避免truncation噪声污染梯度；(b) **软惩罚**：对未截断但较长的回复，引入阶梯式长度惩罚 $r_{\text{length}} = r \cdot \mathbb{I}[L < L_{\text{thresh}}] + r \cdot \alpha^{\lfloor L / \Delta L \rfloor} \cdot \mathbb{I}[L \ge L_{\text{thresh}}]$，鼓励模型在完成任务的前提下尽量精简。这本质是在准确性和经济性之间引入了显式可调的trade-off参数。

### GSPO（Group Sequence Policy Optimization）

GSPO与GRPO的核心分歧在于**重要性采样比率的计算粒度**，这一分歧深刻影响着算法对两类RL场景的适应性。

**Token级比率 vs 序列级比率**：GRPO在token级别计算比率 $r_t = \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{\text{old}}(y_t|x, y_{<t})}$，这意味着策略更新被视为**每个token的独立决策**——如果序列前半段与旧策略一致（$r_t \approx 1$），后半段偏离（$r_t \gg 1$），前半段token几乎不受益/受罚。这适用于**短程决策**场景（如单轮QA的回复措辞调整），此时token之间的依赖相对局部。

GSPO改为在**序列级别**计算重要性比率：

$$r_{\text{seq}}(\theta) = \frac{\prod_{t=1}^{T} \pi_\theta(y_t|x, y_{<t})}{\prod_{t=1}^{T} \pi_{\text{old}}(y_t|x, y_{<t})} = \exp\left( \sum_{t=1}^{T} \log \pi_\theta(y_t|x, y_{<t}) - \log \pi_{\text{old}}(y_t|x, y_{<t}) \right)$$

**序列级比率的物理学直觉**：一条轨迹中的所有token共享同一个比率因子——要么整条轨迹被整体强化，要么被整体抑制。这抓住了Agentic场景的本质特征：工具调用序列中的每一步的价值依赖于整条轨迹的最终成功率，单步无法被独立评估。一条最终成功的轨迹，即使其中某一步看起来"异常"，也应当被整体奖励（因为整体成功证明了该步骤在上下文中的合理性）；反之，一条失败的轨迹，即使大部分步骤看起来正常，也应当被整体抑制。

形式化地，GSPO的损失函数为：

$$\mathcal{L}^{\text{GSPO}} = \min\left( r_{\text{seq}} A_{\text{seq}},\ \text{clip}(r_{\text{seq}}, 1-\epsilon, 1+\epsilon) A_{\text{seq}} \right) - \beta D_{\text{KL}}$$

其中 $A_{\text{seq}}$ 为序列级优势（同样通过组内标准化获得），clip在序列级施加。这意味着**clip保护的是整条策略轨迹不被单步大幅修改**——当一条轨迹的累积比率超出 $[1-\epsilon, 1+\epsilon]$，整条轨迹的梯度被截断，模型在该轨迹上停止更新。这比token级clip更保守，但在agentic场景（该场景下轨迹级稳定性比token级灵活性更重要）中更安全。

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

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta(\cdot|s)}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a)\right]$$

其中优势函数 $A(s, a)$ 仅依赖于当前状态-动作对的即时奖励，无需考虑长期回报的折扣累积。这一简化带来了训练上的巨大便利——梯度估计的方差仅来自单步动作采样，不涉及时间维度的误差传播。

**Contextual Bandit何时足够？** 当任务满足以下条件时，CB建模完全充分：(1) 任务是单轮的，输入和输出之间没有交互循环；(2) 回复质量可以通过最终输出直接评估，无需中间步骤的验证；(3) 不存在工具调用或环境反馈改变后续决策的需求。典型场景包括：单轮问答质量对齐、文本摘要偏好优化、有害内容拒绝等。在这些场景下，PPO/GRPO/DAPO等算法在CB框架内已经表现优异——前文分析的GRPO取代PPO的三大理由（Critic高方差、额外容量开销、参数共享冲突）正是在CB设定下成立的。

**Contextual Bandit何时失效？** CB在以下场景中暴露出根本性的建模不足：

1. **多步工具调用**：当LLM需要先调用搜索引擎获取信息，再基于结果生成回答时，第一次工具调用的结果改变了后续决策的"状态"，这不再是单步决策。CB若强行应用，只能将"prompt + 搜索结果"视为新的单一prompt，但这丢失了"主动选择调用什么工具、如何解读结果"的决策学习。

2. **代码调试循环**：LLM编写代码→执行→观察错误→修改代码→再执行，每次执行结果都是新的状态信息。CB无法建模这种"观察→调整→再观察"的闭环，因为每次工具调用都被拍扁为单步回复的一部分。

3. **长期规划与任务分解**：如"帮我预订机票并安排行程"，涉及多个子任务的串联执行，每个子任务的成功与否影响后续决策路径。CB无法学习到"根据中间结果动态调整计划"的元能力。

在这些场景下强行用CB建模，相当于将多步交互压缩为单步输出，损失了中间反馈提供的丰富学习信号，且无法学习到"根据中间结果调整策略"的元能力——而这恰恰是Agent能力的核心。

### Full MDP：Agentic RL的数学框架

Agentic RL将LLM与环境的交互建模为完整的**有限视界MDP**：

- **状态（State）**：$s_t = (\text{prompt}, h_t, o_t)$，包括原始prompt、工具调用历史 $h_t = (a_1, r_1, a_2, r_2, ..., a_{t-1}, r_{t-1})$ 以及当前环境观测 $o_t$（如代码执行结果、网页内容、API返回值）。状态的维度随 $t$ 增长而持续膨胀——这是MDP场景下Critic训练困难的根源之一。

- **动作（Action）**：$a_t \in \mathcal{A}$，可以是生成文本、调用工具、终止任务等。动作空间 $\mathcal{A}$ 是语言token空间与结构化tool call空间的并集，具有"离散（工具选择）+ 连续（参数生成）"的混合特性。

- **状态转移（Transition）**：$P(s_{t+1}|s_t, a_t)$ 由两部分组成——确定性部分（环境对tool call的响应，如代码执行结果是确定性的）和随机性部分（外部API的不确定性、网络延迟、网页内容的动态变化等）。

- **奖励（Reward）**：$r_t = r(s_t, a_t)$ 或仅在终止状态给出 $r_T$（稀疏奖励）。常见形式包括 $r_T = \mathbb{I}[\text{任务成功}]$（二元）或 $r_T = \text{score}$（连续）。

- **轨迹（Trajectory）**：$\tau = (s_0, a_0, s_1, a_1, ..., s_T)$，长度 $T$ 不定，取决于任务复杂度和模型策略（何时选择"终止"动作）。

在Full MDP下，策略优化的目标变为最大化累积折扣奖励的期望：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

其中 $\gamma \in (0, 1]$ 是折扣因子，控制短期与长期回报的权衡。与CB相比，Full MDP的核心复杂性来自**时间维度的引入**——每一步的决策不仅影响即时奖励，还通过状态转移影响未来的所有决策机会。这也意味着**策略更新不再独立于轨迹**：更新第 $t$ 步的动作概率，会通过改变后续状态的分布间接影响所有 $k > t$ 步的优化。

### 从CB到MDP：算法设计的连锁反应

建模视角从CB升级到MDP，直接导致策略优化算法的设计面临一系列根本性挑战。理解这些挑战是评估PPO/GRPO/DAPO/GSPO在Agentic场景下适用性的关键。

**1. PPO Clip为何在CB中高效、在MDP中受限？**

在Contextual Bandit设定下，PPO的clip操作之所以高效，是因为单步决策不存在时间维度的误差累积：
- 重要性采样比率 $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ 仅涉及当前步的策略比值。
- Clip将其限制在 $[1-\epsilon, 1+\epsilon]$ 后，对整体轨迹的偏差影响是**局部的、可控的**。
- 没有"一步错、步步错"的连锁反应——即使某次更新过于激进，也仅影响当前回复质量，不会通过状态转移"污染"后续决策。

在Full MDP下，clip的保护作用被**时间维度稀释**：
- 即使每步的ratio被clip在 $[1-\epsilon, 1+\epsilon]$ 内，$T$ 步累积的策略偏移可达 $(1+\epsilon)^T$——随轨迹长度指数增长。换言之，**单步clip无法阻止多步累积的策略漂移**。
- 优势函数 $\hat{A}_t$ 的估计需要累积未来奖励，方差随轨迹长度增长（见下文信用分配分析），clip后的梯度信号中信噪比进一步恶化。
- PPO在MDP下的有效使用要求额外的稳定化手段：截断轨迹长度、增大batch size以降低方差、更保守的clip阈值（$\epsilon < 0.1$）。

**2. 信用分配（Credit Assignment）的维度爆炸**

从CB切换到MDP后，信用分配问题从"单步归因"升级为"多步因果推断"：

- 一条Agentic轨迹可能包含20-50步工具调用，最终奖励（任务成功/失败）需要在所有中间步骤间分配"功劳"或"责任"。
- **REINFORCE估计器** $\nabla_\theta J = \mathbb{E}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t]$ 中，回报 $G_t = \sum_{k=t}^T \gamma^{k-t} r_k$ 的方差随 $T$ 线性增长——这是著名的"信用分配方差问题"，在MDP中尤为严重。
- **GAE（广义优势估计）**通过引入Critic的 $V(s_t)$ 作为baseline并在时间维度上做 $\lambda$ 指数衰减加权，将方差从 $O(T)$ 压缩至 $O(T \cdot \lambda^T)$。但当 $T$ 较大时（如Agentic场景下 $T=20+$），即使 $\lambda=0.95$，衰减因子 $\lambda^{20} \approx 0.36$——长距离的因果信号仍然无法有效传播。

更深层的问题在于**时间维度的因果解耦**：假设一条代码修复轨迹包含20步操作，第5步引入了一个"隐式bug"（修改了配置文件），第15步才因该bug导致测试失败。最终奖励信号反馈到第5步需要跨越10步的梯度传播，而在这10步中间，模型又做了大量"正确"的操作——这些操作的梯度信号被**错误地"污染"**：既承担了失败的责任（因为最终失败了），又被鼓励（因为中间步骤本身是正确的）。标准GAE的指数衰减无法区分这种非马尔可夫的因果结构。

**3. GRPO组内相对优势的MDP适配**

GRPO的核心创新——用组内标准化替代Critic的价值估计——在MDP场景下需要仔细重新设计：

- **轨迹级优势（最直接）**：对同一prompt，采样 $G$ 条完整轨迹 $\tau_1, ..., \tau_G$，计算每条轨迹的总奖励 $R(\tau_i) = \sum_t r_t^{(i)}$，然后进行组内标准化 $A(\tau_i) = \frac{R(\tau_i) - \text{mean}(R)}{\text{std}(R)}$。将轨迹级优势均匀分配给轨迹内的每一步。优点是实现简单；缺点是丢失了步级区分度——轨迹中每一步都被赋予相同优势，信用分配的粒度粗糙。

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

- **PPO中优势估计的方差爆炸**：在MDP下，优势函数 $\hat{A}_t$ 的方差随轨迹长度呈 $O(T)$ 增长（REINFORCE）或 $O(T \cdot \lambda^T)$（GAE with $\lambda < 1$）。对于 $T=30$ 的Agentic轨迹，即使 $\lambda=0.95$，前期步骤的优势估计几乎退化为噪声——信噪比过低以至于无法驱动有意义的梯度更新。

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
| **DPO** | 离线偏好优化，无需在线采样 | 不直接适用——DPO假设单轮 $(y_w, y_l)$ 偏好对，无法建模多步轨迹 | 无法处理状态转移，偏好数据的构造在MDP下定义模糊 |
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
\alpha_t = \alpha_{\text{start}} \cdot \gamma^{t/T} + \alpha_{\text{end}} \cdot (1 - \gamma^{t/T})
$$

其中 $\alpha_t$ 为第 $t$ 步的蒸馏损失权重，从高初值 $\alpha_{\text{start}}$（如 0.8~0.9，蒸馏主导）逐步衰减至终值 $\alpha_{\text{end}}$（如 0.1~0.2，RL 主导），$\gamma$ 控制衰减速率，$T$ 为总训练步数。这种渐进式的"松绑"策略在实践中被证明能显著降低 RL 训练的失败率，尤其适合从 SFT 模型初始化起的端到端训练。

### KL 正则化的等价性分析

OPD 的蒸馏损失与 PPO 的 KL 惩罚项在数学形式上高度相似，但两者存在关键差异。

**PPO 的 KL 惩罚**：在奖励函数中减去当前策略 $\pi_\theta$ 与参考策略 $\pi_{\text{ref}}$ 之间的 KL 散度：

$$
r_{\text{KL-penalized}} = r(s, a) - \beta \cdot D_{\text{KL}}(\pi_\theta(\cdot|s) \parallel \pi_{\text{ref}}(\cdot|s))
$$

其中 $\pi_{\text{ref}}$ 通常是训练开始前的 SFT 模型（或上一轮迭代的快照），**保持冻结不变**。其作用是防止策略偏离初始分布太远，本质上是一个"回归原点"的约束。

**OPD 的蒸馏损失**：直接在损失函数中加入学生策略 $\pi_S$ 与教师策略 $\pi_T$ 之间的 KL 散度：

$$
\mathcal{L}_{\text{distill}} = D_{\text{KL}}(\pi_T(\cdot|s) \parallel \pi_S(\cdot|s))
$$

这里的核心区别在于：**教师 $\pi_T$ 是一个更强的、动态更新的模型**（如经过完整 RL 训练的 R1），其输出分布本身就是高质量的优化目标，而非仅仅是一个"不回退"的锚点。

两者对比：

| 特性 | PPO KL 惩罚 | OPD 蒸馏损失 |
|------|-----------|------------|
| **参考模型** | $\pi_{\text{ref}}$（冻结旧策略） | $\pi_T$（更强的教师模型） |
| **目标方向** | 约束：不要离起点太远 | 引导：朝更好的方向移动 |
| **模型能力** | 通常 ≤ 当前策略 | 通常 > 当前策略 |
| **更新方式** | 训练全程冻结 | 可静态也可迭代更新 |

**何时冗余、何时互补？** 如果教师模型 $\pi_T$ 本身就远强于学生，蒸馏损失已经天然包含了"不退化"的约束（往教师方向靠拢本身就是一种正则化），此时再叠加 PPO 式的 ref-model KL 惩罚是**冗余的**，甚至可能因过度约束拖慢优化速度。但当教师与学生能力接近（如同规模的 online distillation），蒸馏信号的引导力不足时，额外的 ref-model KL 惩罚作为**互补正则化**有助于防止策略在低置信度区域剧烈震荡。

### OPD 损失函数构成

On-Policy Distillation 的完整损失函数由策略梯度损失和蒸馏损失加权组合而成：

```python
def opd_loss(
    student_logits,       # 学生模型输出的 logits
    teacher_logits,       # 教师模型输出的 logits（同 batch 在线生成）
    old_log_probs,        # 采样时学生模型的 log prob（用于重要性采样）
    advantages,           # 优势函数（GAE 或 GRPO 组内标准化）
    actions,              # 实际采样的 token 序列
    alpha=0.5,            # 蒸馏损失权重（可随训练步数退火）
    clip_eps=0.2,         # PPO clip 阈值
    kl_beta=0.1,          # （可选）额外 ref-model KL 惩罚系数
):
    # ---- 1. 策略梯度损失（PPO-clipped） ----
    new_log_probs = F.log_softmax(student_logits, dim=-1)
    action_log_probs = new_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    ratio = (action_log_probs - old_log_probs).exp()  # 重要性采样比率
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # ---- 2. 蒸馏损失（教师→学生 KL 散度） ----
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    distill_loss = F.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean"
    )

    # ---- 3. （可选）额外 ref-model KL 惩罚 ----
    if ref_logits is not None:
        ref_probs = F.softmax(ref_logits / temperature, dim=-1)
        ref_kl = F.kl_div(student_log_probs, ref_probs, reduction="batchmean")
    else:
        ref_kl = 0.0

    # ---- 4. 组合损失 ----
    # alpha 采用退火调度：从 ~0.9（蒸馏主导）逐步衰减至 ~0.1（RL 主导）
    total_loss = (1 - alpha) * policy_loss + alpha * distill_loss + kl_beta * ref_kl

    return total_loss
```

关键设计要点：
- **$\alpha$ 退火**：`alpha` 不是固定值，而是随训练步数按指数衰减。早期 $\alpha$ 大，模型主要模仿教师，RL 信号仅作微调；后期 $\alpha$ 小，模型开始自主探索，教师仅提供兜底正则化。
- **temperature 共享**：蒸馏部分的 softmax 温度通常与教师推理时一致，保证分布平滑度匹配。
- **ref_kl 可选**：当教师远强于学生时，`ref_kl` 通常置零以免冗余约束；当教师与学生能力接近时，保留 `ref_kl` 作为额外稳定项。

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
