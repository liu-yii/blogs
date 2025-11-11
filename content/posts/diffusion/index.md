---
date: 2025-06-16T14:45:29+08:00
draft: false
title: 'Diffusion'
---

## Preliminary Knowledge

### 条件概率公式

条件概率的一般形式：

$$
P(A,B,C)=P(C|B,A)P(B,A)=P(C|B,A)P(B|A)P(A)
$$

$$
P(B,C|A)=P(C|B,A)P(B|A)
$$

马尔可夫条件：下一状态的概率分布只能由当前状态决定，与前面的状态无关。

$$
P(A,B,C)=P(C|B)P(B|A)P(A)
$$

$$
P(B,C|A)=P(C|B)P(B|A)
$$

### KL散度

KL散度是衡量两个概率分布之间差异的一种度量方法，它衡量了从一个分布到另一个分布所需的额外信息。KL散度的定义是建立在熵 Entropy 的基础上的，熵的定义如下：

$$
H(X)=-\sum_{i=1}^{n}p_i\log p_i
$$

规定当 $p_i=0$ 时，$p_i\log p_i=0$

$$
H(p,q)=-\sum_{i=1}^{n}p(x)\log q(x)
$$

在信息论中，交叉熵可认为是对预测分布 $q(x)$ 用真实分布 $p(x)$ 来进行编码时所需要的信息量大小。因此我们可以通过交叉熵和信息熵来推导相对熵（KL散度）：

$$
\begin{align}
D_{KL}(p||q)&=H(p,q)-H(p) \\\\
&=-\sum_{i=1}^{n}p(x)\log q(x)+\sum_{i=1}^{n}p(x)\log p(x) \\\\
&=-\sum_{i=1}^{n}p(x)\log \frac{q(x)}{p(x)}
\end{align}
$$

KL散度的特点：

1. 非对称性：$D_{KL}(p||q)\neq D_{KL}(q||p)$  
2. 非负性：$D_{KL}(p||q)\geq 0$

$$
\begin{align}
p(x)&=\frac{1}{\sqrt{2\pi}\sigma_1}\exp\left({-\frac{(x-\mu_1)^2}{2\sigma_1^2}}\right) \\\\
q(x)&=\frac{1}{\sqrt{2\pi}\sigma_2}\exp\left({-\frac{(x-\mu_1)^2}{2\sigma_2^2}}\right)
\end{align}
$$

$$
\begin{align}
\int p(x)\log(p(x))dx &= -\frac{1}{2}\left[1+\log(2\pi\sigma_1^2)\right] \\\\
\int p(x)\log(q(x))dx &= -\frac{1}{2}\log(2\pi\sigma_2^2)-\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}
\end{align}
$$

$$
D_{KL}(p||q)=\log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac{1}{2}
$$

### 高斯分布的重参数化

若希望从高斯分布中采样，我们可以使用标准正态分布 $\mathcal{N}(0,1)$ 来采样 $z$，然后通过重参数化 $\sigma*z+\mu$ 的方式将其转换为高斯分布 $N(\mu,\sigma^2)$。

这样做的好处在于将随机性转移到了 $z$ 这个常量上，使得采样过程梯度可传播，从而可以使用梯度下降等优化算法进行训练。

---

## VAE 与多层 VAE

### 单层 VAE

![VAE 结构图](img/VAE.png)

$$
p(x)=\int_{z} p(x, z)
$$

$$
p(x)=\int_{z} p_{\theta}(x|z)p(z)
$$

$$
p(x)=\int_{z} q_{\phi}(z|x)\frac{p_{\theta}(x|z)p(z)}{q_{\phi}(z|x)}
$$

$$
\begin{align}
\log p(x) &= \log \int_{z} q_{\phi}(z|x)\frac{p_{\theta}(x|z)p(z)}{q_{\phi}(z|x)} \\
\nonumber
&= \log \mathbb{E}_{z\sim q_{\phi}(z|x)}\left[\frac{p_{\theta}(x|z)p(z)}{q_{\phi}(z|x)}\right]
\end{align}
$$

$$
\begin{align}
\log p(x) &\ge \mathbb{E}_{z\sim q_{\phi}(z|x)}\left[\log \frac{p_{\theta}(x|z)p(z)}{q_{\phi}(z|x)}\right] \\
\nonumber
&= \mathbb{E}_{z\sim q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - D_{KL}(q_{\phi}(z|x) || p(z))
\end{align}
$$



第一项为重建项，第二项为正则化项。

### 多层 VAE

（待更新）

---

## Diffusion Models
主要参考了[Lil's Log](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#quick-summary)
### 前向扩散过程（加噪）
Given a data point sampled form a real data distribution $x_0 \sim q(x)$, we  define a **forward diffusion process**: we add small amount of Gaussian noise to the sample in $T$ steps, producing a sequence of noisy samples $x_1,...,x_T$. The step sizes are controlled by a variance schedule $\{\beta_{t}\in(0,1)\}_{t=1}^{T}$.

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)\\
q(x_{1:T}|x_0) = \prod_{t=1}^{T}q(x_t|x_{t-1})
$$

*这里的前向过程可以视为马尔可夫过程，即当前状态$x_t$只与上一时刻的状态$x_{t-1}$有关。在给定上一状态$x_{t-1}$的条件下，获取第$t$步样本$x_t$的概率分布q(x_t|x_{t-1})。$\mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ 表示给定上一步的状态$x_{t-1}$， $x_{t}$是一个以$\sqrt{1-\beta_t}x_{t-1}$为均值, $\beta_t I$为协方差的高斯分布的随机变量.*

*也就是说：$x_{t} = \sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_{t}}\epsilon_{t-1}, \epsilon_{t-1}\in\mathcal{N}(0,I)$*

*因此，当步数$t$逐渐增大时，采样数据$x_0$会逐渐趋向于纯高斯噪声*
![VAE 结构图](img/diffusion_process.png)

At any arbitrary time step $t$, we can sample $x_t$ in above process. Let $\alpha_{t} = 1-\beta_{t}$ and $\bar{\alpha}_{t} = \prod_{i=1}^{t}\alpha_{i}$:

$$
\begin{align}

x_{t} &= \sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_{t}}\epsilon_{t} \\
\nonumber
&=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t}\alpha_{t-1}}\epsilon_{t-2} \\
\nonumber
&=...\\
\nonumber
&=\sqrt{\bar{\alpha}_{t}}x_{0}+\sqrt{1-\bar{\alpha}_{t}}\epsilon \\
q(x_t|x_0) &= \mathcal{N}(x_t; \sqrt{\bar{\alpha}_{t}}x_{0}, (1-\bar{\alpha}_{t})I)
\end{align}
$$


### 反向扩散过程（去噪）
If we can reverse the forward process and sample from $q(x_{t-1}|x_t)$, we will be able to recreate the true sample form a Gaussian noise input, $x_{T} \sim \mathcal{N}(0,I)$. Unfortunately, we cannot easily estimate $q(x_{t-1}|x_t)$ because it needs to use the entire dataset and therefore we need to learn a model $p_{\theta}$ to approximate these conditional probabilities in order to run the *reverse diffusion process*.
$$
\begin{align}
p_{\theta}(x_{0:T}) &= p(x_T)\prod_{i=1}^{t}p_{\theta}(x_{t-1}|x_t)\\
p_{\theta}(x_{t-1}|x_t) &= \mathcal{N}(x_{t-1};\mu_{\theta}(x_t,t), \Sigma_{\theta}(x_t,t))
\end{align}
$$

*如何学习这一过程呢？最简单的方法就是最小化$x_{t-1}$和$p_{\theta}(x_{t-1}|x_t)$之间的欧式距离*：
$$
||x_{t-1}-p_{\theta}(x_{t-1}|x_t)||^{2}
$$
*继续细化这一过程，将前向过程可以改写为$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}(x_t-\sqrt{1-\alpha_{t}}\epsilon_{t})$，因此模型$p_{\theta}$可以设计为*：
$$
p_{\theta}(x_{t-1}|x_t) = \frac{1}{\sqrt{\alpha_t}}(x_t-\sqrt{1-\alpha_{t}}\epsilon_{\theta}(x_t,t))
$$
*代入训练损失函数*：
$$
||x_{t-1}-p_{\theta}(x_{t-1}|x_t)||^{2} = \frac{\beta_{t}}{\alpha_t}||\epsilon_{t}-\epsilon_{\theta}(x_t,t)||^{2}
$$
*忽略常数系数$\frac{\beta_{t}}{\alpha_t}$，然后代入$x_{t} = \sqrt{\bar{\alpha}_{t}}x_{0}+\sqrt{1-\bar{\alpha}_{t}}\epsilon$，最终得到的损失函数*：
$$
||\epsilon_{t}-\epsilon_{\theta}(\sqrt{\bar{\alpha}_{t}}x_{0}+\sqrt{1-\bar{\alpha}_{t}}\epsilon_{t},t)||^{2}
$$


## Conditioned Generation
### Classifier Guided Diffusion 