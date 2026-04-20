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
- CoT Prompting：(...)
- SFT：(...)
- RL：(...)

Efficient Reasoning方法：(...)


## Pocliy Optimization
### PPO（Proximal Policy Optimization）
PPO是一种策略优化算法，在进行持续优化策略的同时，严格限制每次策略更新的幅度，从而保证训练的稳定。

PPO采用Actor-Critic架构，训练中包括四个模型：Actor模型，Critic模型，Reward模型以及Ref模型。PPO计算新旧策略在相同状态下输出动作概率的比值（重要性采样比率），并结合优势函数$A_t$对Actor模型进行优化。

优势函数 $A_t$ 以Critic模型估计的奖励为baseline，通过**GAE（广义优势估计）**，计算未来每一步的TD误差然后进行指数衰减的加权求和得到该步的优势函数，并且在每一步的奖励函数中添加与ref模型的**KL散度惩罚项**，防止Actor模型偏离Ref模型太远。

PPO的关键优化是**clip操作**，将新旧策略的概率比值强行限制在 $[1-\epsilon, 1+\epsilon]$ 的范围内，控制策略更新的步长，**防止模型在单步训练中更新过快导致策略崩溃（破坏旧策略的分布，陷入局部最优）**。
### DPO（Direct Policy Optimization）
DPO的核心思想是绕过传统的reward model和复杂的PPO强化学习阶段，直接使用人类偏好数据对语言模型进行优化。

它通过损失函数拉大模型生成chosen回复和rejected回复的概率差，避免了PPO训练时的在线采样成本以及Actor-Critic架构的崩溃风险。

### GRPO（Group Relative Policy Optimization）
GRPO针对同一个输入Prompt，让Actor模型一次性采样生成一组（Group）多个不同的回复，然后通过Reward Model或基于规则的验证器对这组回复分别计算奖励得分。

接着，GRPO直接在这组得分内部进行统计，算出平均值和标准差，以此对各个回复的得分进行标准化，将原本需要Critic预测的复杂状态价值替换成了计算成本极低的组内相对优势 $A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$。

最后，GRPO利用这个相对优势值，结合参考模型的KL散度惩罚和clip操作来更新Actor模型。

### DAPO（Decoupled clip and Dynamic sAmpling Policy Optimization）
DAPO相较于GRPO有四个改进：
1. higher clip：DAPO提高了clip的upper阈值，提高策略的多样性
2. dynamic sampling：为了避免遇到较难的问题或者较简单的问题GRPO组内相对优势较小的情况（Zero-Variance），DAPO提出动态采样机制，强制组内答案的奖励方差不为0
3. token level策略梯度损失：GRPO采用seq-level的梯度损失，对于高质量长序列的回答，其梯度比率较小，阻碍了模型学习其中的逻辑，而对于存在reward hacking的输出，样本级的损失无法有效地惩罚这种不良模式。DAPO将GRPO原先在句子级别求平均的损失计算改为了Token级别的直接聚合，解决了高质量的长序列推理步骤在梯度更新时梯度权重被过度稀释的偏差问题。
4. Overlong reward shaping：在计算损失时直接剔除被强制截断的噪声样本，并引入阶梯式的长度软惩罚，引导模型精准控制输出长度。
