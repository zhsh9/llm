# LSTM记忆元$c_t$的计算过程

## 1. 初始公式

LSTM中记忆元更新的基本公式是：

$$c_t = i_t \odot \tilde{c}_t + f_t \odot c_{t-1}$$

其中：
- $c_t$是当前时刻的记忆元
- $i_t$是输入门
- $\tilde{c}_t$是候选记忆元
- $f_t$是遗忘门
- $\odot$表示Hadamard积（元素wise乘法）

## 2. 递归展开

我们可以将$c_{t-1}$继续展开：

$$
\begin{aligned}
c_t &= i_t \odot \tilde{c}_t + f_t \odot c_{t-1} \\
&= i_t \odot \tilde{c}_t + f_t \odot (i_{t-1} \odot \tilde{c}_{t-1} + f_{t-1} \odot c_{t-2}) \\
&= i_t \odot \tilde{c}_t + f_t \odot i_{t-1} \odot \tilde{c}_{t-1} + f_t \odot f_{t-1} \odot c_{t-2}
\end{aligned}
$$

继续这个过程，我们得到：

$$
\begin{aligned}
c_t &= i_t \odot \tilde{c}_t \\
&+ f_t \odot i_{t-1} \odot \tilde{c}_{t-1} \\
&+ f_t \odot f_{t-1} \odot i_{t-2} \odot \tilde{c}_{t-2} \\
&+ f_t \odot f_{t-1} \odot f_{t-2} \odot i_{t-3} \odot \tilde{c}_{t-3} \\
&+ \cdots
\end{aligned}
$$

## 3. 求和表示

观察上面的展开式，我们可以将其写成求和的形式：

$$c_t = \sum_{i=1}^t (\text{权重项}) \odot \tilde{c}_i$$

## 4. 权重项推导

分析每个$\tilde{c}_i$前面的系数，我们可以得到权重项的通用形式：

$$\text{权重项} = \left( \prod_{j=i+1}^t f_j \right) \odot i_i$$

## 5. 最终公式

将权重项代入求和表达式，我们得到最终的公式：

$$c_t = \sum_{i=1}^t \left( \prod_{j=i+1}^t f_j \odot i_i \right) \odot \tilde{c}_i$$

## 6. 简化表示

为了简化表示，我们可以定义：

$$w_i^t = \prod_{j=i+1}^t f_j \odot i_i$$

那么最终的公式可以写成：

$$c_t = \sum_{i=1}^t w_i^t \odot \tilde{c}_i$$

这个公式表示了LSTM如何在长时间序列中累积和更新信息。每个时间步的候选状态$\tilde{c}_i$都对当前状态$c_t$有贡献，但贡献的大小由过去的遗忘门和输入门共同决定，体现在权重$w_i^t$中。