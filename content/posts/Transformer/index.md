---
date: '2025-12-19T18:36:19+08:00'
draft: false
title: 'Transformer基础'
---

## 位置编码
在这里记录一下对Transformer中位置编码的理解，主要参考了苏神[Transformer升级之路](https://www.spaces.ac.cn/archives/8231)中的部分内容，论证过程不一定非常严谨，主要是用自己能够理解的角度去看待位置编码。
### 为什么需要位置编码？
假设模型的输入是 $(\cdots, x_m, \cdots, x_n, \cdots)$，如果不加入位置编码，由于Attention模型f是全对称的，即对于任意的$m,n$，有：
$$
f(\cdots, x_m, \cdots, x_n, \cdots) = f(\cdots, x_n, \cdots, x_m, \cdots)
$$

这也意味着在模型看来，倒序的输入和正序的输入是一致的，但是对于语言输入来说，顺序是非常重要的，因此需要加入位置编码 $p_m, p_n$来引入位置信息，一般的做法是直接相加，即：
$$
\hat{f}(\cdots, x_m, \cdots, x_n, \cdots) = f(\cdots, x_m+p_m, \cdots, x_n+p_n, \cdots)
$$

我们希望，一个好的位置编码不仅能够提供绝对位置信息，还能提供相对的位置信息，那么仅仅考虑$x_m$和$x_n$的Attention计算(向量的内积)，$x_m x_n + p_m x_n + p_n x_m + p_m p_n$，绝对位置信息通过第2、3项提供，相对位置信息通过第4项提供。

### Sinusoidal位置编码
我们希望$\left \langle p_m, p_n  \right \rangle$ (两个位置编码的内积)能够表达相对位置信息，即存在某个函数$g$使得

$$
\left \langle p_m, p_n  \right \rangle = g(m-n)
$$
$p_m, p_n$为$d$维向量，我们从最简单的$d=2$入手，对于二维向量，我们将向量$[x,y]$表示为复数形式$x+iy$，另外假设存在复数$q_{m-n}$，使得：
$$
p\_m p^{*}\_n = q\_{m-n}
$$

这里$p^{*}\_n$为$p\_n$的共轭复数，然后用复数的指数形式求解，即设$p\_m = r\_m e^{i\phi\_{m}}$, $p\_n = r\_n e^{i\phi\_{n}}$, $q_{m-n} = R_{m-n} e^{i\Phi\_{m-n}}$，具体的求解过程参照[Transformer升级之路：1、Sinusoidal位置编码追根溯源](https://www.spaces.ac.cn/archives/8231)。
最终，可以得到通解为：
$$
p_m = e^{im\theta}  \Leftrightarrow  p_m = \binom{\cos m\theta}{\sin m\theta} 
$$
扩展到更高维得到：
$$
p_m = 
\begin{pmatrix}
e^{im\theta\_{0}} \\\\
e^{im\theta\_{1}} \\\\
\cdots \\\\
e^{im\theta\_{d/2-1}} 
\end{pmatrix}
\Leftrightarrow 
p_m = \begin{pmatrix}
\cos m\theta\_{0} \\\\ 
\sin m\theta\_{0} \\\\ 
\cos m\theta\_{1} \\\\ 
\sin m\theta\_{1} \\\\ 
\cdots \\\\ 
\cdots \\\\ 
\cos m\theta\_{d/2-1} \\\\ 
\sin m\theta\_{d/2-1} \\\\ 
\end{pmatrix}
$$

取$\theta_{i} = 10000^{-2i/d}$ 即可以得到传统的Sinusoidal位置编码

### RoPE位置编码
在RoPE中，通过以下运算来给$q,k$添加绝对位置信息：
$$
q_m = f(q,m), k_n = f(k,n)
$$
同时我们也希望$q,k$的内积带有相对位置关系:
$$
\left \langle f(q,m), f(k,n)  \right \rangle = g(q, k, m-n)
$$
那么和上面的思路一样，我们也需要求解该方程的一个特殊解。同时我们加入初始条件$f(q,0) = q, f(k,0) = k$。
具体的过程也是通过二维向量的复数形式来进行求解，参照[Transformer升级之路：2、博采众长的旋转式位置编码](https://www.spaces.ac.cn/archives/8265)。
最终可以得到:
$$
f(q,m) = R_{f}(q,m)e^{i\Theta(q,m)} = \left \| q \right \| e^{i(\Theta(q)+m\theta)} = qe^{im\theta}
$$
可以写成矩阵形式：
$$
f(q,m) = \begin{pmatrix}
  \cos m\theta & -\sin m\theta \\\\
  \sin m\theta & \cos m\theta
\end{pmatrix}
\begin{pmatrix}
  q_0 \\
  q_1
\end{pmatrix}
$$
扩展到d维
$$
\begin{pmatrix}
  \cos m\theta\_{0} & -\sin m\theta\_{0} & 0 & 0 & \cdots & 0 & 0 \\\\
  \sin m\theta\_{0} & \cos m\theta\_{0} & 0 & 0 & \cdots & 0 & 0 \\\\
  0 & 0 & \cos m\theta\_{1} & -\sin m\theta\_{1} &  \cdots & 0 & 0 \\\\
  0 & 0 & \sin m\theta\_{1} & \cos m\theta\_{1} &  \cdots & 0 & 0 \\\\
  \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots \\\\
  \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots \\\\
  0 & 0 & 0 & 0 & \cdots & \cos m\theta\_{d/2-1} & -\sin m\theta\_{d/2-1} \\\\
  0 & 0 & 0 & 0 & \cdots & \sin m\theta\_{d/2-1} & \cos m\theta\_{d/2-1} \\\\
\end{pmatrix}
\begin{pmatrix}
q_0 \\\\
q_1 \\\\
q_2 \\\\
q_3 \\\\
\cdots \\\\
q_{d-2} \\\\
q_{d-1}
\end{pmatrix}
$$

即给位置为 $m$ 的向量 $q$ 乘上矩阵 $R_m$, 给位置为 $n$ 的向量 $k$ 乘上矩阵 $R_{n}$，然后用变换后的 $Q,K$ 序列做Attention计算，Attention中就自动包含相对位置信息了。

由于矩阵 $R_{n}$的稀疏性，在具体实现的过程中可以用下面的方式：
$$
\begin{pmatrix}
  q_{0} \\\\
  q_{1} \\\\
  q_{2} \\\\
  q_{3} \\\\
  \cdots \\\\
  q_{d-2} \\\\
  q_{d-1} \\\\
\end{pmatrix} \otimes
\begin{pmatrix}
  \cos m\theta\_{0} \\\\
  \cos m\theta\_{0} \\\\
  \cos m\theta\_{1} \\\\
  \cos m\theta\_{1} \\\\
  \cdots \\\\
  \cos m\theta\_{d/2-1} \\\\
  \cos m\theta\_{d/2-1} \\\\
\end{pmatrix}+
\begin{pmatrix}
  -q_{1} \\\\
  q_{0} \\\\
  -q_{3} \\\\
  q_{2} \\\\
  \cdots \\\\
  q_{d-1} \\\\
  q_{d-2} \\\\
\end{pmatrix} \otimes
\begin{pmatrix}
  \sin m\theta\_{0} \\\\
  \sin m\theta\_{0} \\\\
  \sin m\theta\_{1} \\\\
  \sin m\theta\_{1} \\\\
  \cdots \\\\
  \sin m\theta\_{d/2-1} \\\\
  \sin m\theta\_{d/2-1} \\\\
\end{pmatrix}
$$
