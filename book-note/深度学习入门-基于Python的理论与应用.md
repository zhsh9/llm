# 深度学习入门: 基于Python的理论与实现

## Python Crash Course

- NumPy
- Matplotlib

## 感知机 Perceptron

- 单层感知机可以实现与门、与非门、或门，局限在于无法实现异或门，因为其线性性
- 多层感知机具有非线性，可以解决异或门的问题
- 使用了非线性的sigmoid函数的2层感知机可以表示任何函数
- 感知机的权重和偏置是人工设定的，所以会引出神经网络，自动从训练数据学习到合适的权重参数
- 感知机使用跃迁函数，神经网络使用激活函数

## 神经网络

### 激活函数

Sigmoid Activation Function:

$$
h(x) = \frac{1}{1+exp(-x)}
$$

Sigmoid函数和跃迁函数的比较:  
- Sigmoid函数更平滑
- 两者都是非线性函数，神经网络必须使用非线性函数

### 网络传递=矩阵运算

![Matrix_Multi](./深度学习入门.assets/1.png)

### 输出层的设计

- 神经网络目标解决的问题：回归问题、分类问题
- 输出层: 回归问题使用恒等函数，分类问题使用 softmax 函数
- 分类问题中，输出层的神经元的数量设置为要分类的类别数

Softmax Function:

$$
y_k = \frac{exp(a_k)}{\sum_{i=1}^n exp(a_i)}
$$

Softmax 函数的缺陷:  
- 因为有指数运算，所以可能存在溢出问题（inf）
- 在进行 softmax 的指数函数的运算时，加上(或者减去) 某个常数并不会改变运算的结果
- 为了防止溢出，一般会使用输入信号中的最大值

$$
y_k = \frac{\exp(a_k)}{\sum_{i=1}^n \exp(a_i)} = \frac{C \exp(a_k)}{C \sum_{i=1}^n \exp(a_i)}
$$

$$
= \frac{\exp(a_k + \log C)}{\sum_{i=1}^n \exp(a_i + \log C)}
= \frac{\exp(a_k + C')}{\sum_{i=1}^n \exp(a_i + C')}
$$

解决机器学习问题可以被划分为 学习 和 推理 两个阶段:  
- 学习和反向传播有关，推理只和前向传播有关
- 推理阶段一般不使用 softmax
- 因为 softmax 是和神经网络的学习有关

### 数据预处理

- normalization
- standardization

## 神经网络的学习

模型学习指的是：以最小化损失函数为基准，利用误差反向传播法(从梯度下降法引申而来)，从训练数据中自动获取最优权重参数的过程。

图像分类，机器学习算法：

- 从图像中提取特征量，把图像嵌入转换为向量，用 SVM KNN 等分类器训练
- 计算机视觉领域，常用的特征量包括 SIFT、SURF 和 HOG 等

机器学习图像的特点：

- 可解释性强，人为可设定设定的特征量
- 模型适用性狭窄，不同类型的图像需要人为设计专门的特征量

深度学习算法的特点：

- 可解释性低，不需要人为设计特征量
- 神经网络学习提取特征的过程
- end-to-end machine learning

损失函数：

- 均方误差: $E=\frac{1}{2}\sum_{k}(y_k-t_k)^2$
- 交叉熵: $E=-\sum_k t_k\log y_k$

微积分的内容：

- 数值微分(Numerical Differentiation)
- 导数(Derivative)
- 偏导数(Partial Derivative)
- 由全部变量的偏导数汇总得到的为梯度(Gradient)

函数的极小值、最小值以及被称为鞍点(saddle point)的地方，梯度为 0：

- 极小值是局部最小值，也就是限定在某个范围内的最小值
- 鞍点是从某个方向上看是极大值，从另一个方向上看则是极小值的点

根据目的是寻找最小值还是最大值，梯度法的叫法有所不同。严格地讲，寻找最小值的梯度法称为梯度下降法(gradient descent method)，寻找最大值的梯度法称为梯度上升法(gradient ascent method)。但是通过反转损失函数的符号，求最小值的问题和求最大值的问题会变成相同的问题，因此“下降”还是“上升”的差异本质上并不重要。一般来说，神经网络(深度学习)中，梯度法主要是指梯度下降法。

深度学习，LR 学习率设置小技巧：

- 先放大 LR 到训练过程抖动，然后逐步减小
- 或者使用学习率查找器

随机梯度下降法(stochastic gradient descent)
```
设置学习率 learning_rate, 迭代次数 num_iterations, 批量大小 batch_size
随机初始化模型参数 weights 和 biases

for i = 1 to num_iterations do:
    在训练数据中随机选择一个批量(batch)的样本
    
    for 每个批量中的样本 (x, y) do:
        # 前向传播
        y_pred = 模型预测(x, weights, biases)
        
        # 计算损失
        loss = 损失函数(y_pred, y)
        
        # 反向传播
        计算损失函数关于权重 weights 和偏置 biases 的梯度 gradients
        
        # 更新参数
        weights = weights - learning_rate * gradients_weights
        biases = biases - learning_rate * gradients_biases
    end for
    
    # 可选: 打印训练进度或在验证集上评估模型性能
    if i % print_interval == 0:
        在验证集上计算模型的损失和准确率
        打印当前迭代次数, 训练损失, 验证损失和准确率
    end if
end for

return 训练后的模型参数 weights 和 biases
```

神经网络进行的处理过程有学习(Train)和推理(Inference)，推理阶段通常不使用 softmax 函数。

## 误差反向传播法

链式法则是关于复合函数的导数的性质:  
如果某个函数由复合函数表示，则该复合函数的导数可以用构成复合函数的各个函数的导数的乘积表示。

### Back Propagation

$$
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial t} \frac{\partial t}{\partial x}
$$

![Chain Rule](./深度学习入门.assets/2.png)

误差反向传播公式:

$$
\frac{\partial E}{\partial w_{jk}} = \frac{\partial E}{\partial o_j} \frac{\partial o_j}{\partial net_j} \frac{\partial net_j}{\partial w_{jk}}
$$

$$
\frac{\partial E}{\partial w_{jk}} = \delta_j \cdot o_k
$$

其中:
- $\frac{\partial E}{\partial w_{jk}}$ 表示误差 $E$ 对权重 $w_{jk}$ 的偏导数
- $\frac{\partial E}{\partial o_j}$ 表示误差 $E$ 对神经元 $j$ 的输出 $o_j$ 的偏导数
- $\frac{\partial o_j}{\partial net_j}$ 表示神经元 $j$ 的输出 $o_j$ 对其净输入 $net_j$ 的偏导数,即激活函数的导数
- $\frac{\partial net_j}{\partial w_{jk}}$ 表示神经元 $j$ 的净输入 $net_j$ 对权重 $w_{jk}$ 的偏导数,即神经元 $k$ 的输出 $o_k$
- $\delta_j$ 表示神经元 $j$ 的误差项,即 $\frac{\partial E}{\partial o_j} \frac{\partial o_j}{\partial net_j}$

通过应用链式法则,可以将误差从输出层反向传播到输入层,并计算每个权重的梯度。然后,使用梯度下降等优化算法更新权重以最小化误差。

反向传播算法的步骤如下:
1. 前向传播:将输入数据通过神经网络计算输出
2. 计算输出层的误差项 $\delta$
3. 反向传播误差:从输出层开始,通过链式法则计算每个神经元的误差项 $\delta$
4. 计算权重的梯度:使用误差项 $\delta$ 和相应的输入计算每个权重的梯度
5. 更新权重:使用梯度下降等优化算法更新权重以最小化误差
6. 重复步骤 1-5,直到满足停止条件(如达到最大迭代次数或误差低于阈值)

反向传播误差的核心公式是误差项 $\delta$ 的计算公式。对于不同类型的神经元,误差项的计算公式略有不同。

1. 对于输出层神经元,误差项 $\delta_j$ 的计算公式为:

$$
\delta_j = \frac{\partial E}{\partial o_j} \frac{\partial o_j}{\partial net_j}
$$

其中:
- $\frac{\partial E}{\partial o_j}$ 表示误差 $E$ 对输出层神经元 $j$ 的输出 $o_j$ 的偏导数
- $\frac{\partial o_j}{\partial net_j}$ 表示输出层神经元 $j$ 的输出 $o_j$ 对其净输入 $net_j$ 的偏导数,即激活函数的导数

2. 对于隐藏层神经元,误差项 $\delta_j$ 的计算公式为:

$$
\delta_j = \left(\sum_{k \in downstream} w_{kj} \delta_k\right) \frac{\partial o_j}{\partial net_j}
$$

其中:
- $\sum_{k \in downstream} w_{kj} \delta_k$ 表示下一层(下游)所有神经元的误差项 $\delta_k$ 乘以它们对应的权重 $w_{kj}$ 的和
- $\frac{\partial o_j}{\partial net_j}$ 表示隐藏层神经元 $j$ 的输出 $o_j$ 对其净输入 $net_j$ 的偏导数,即激活函数的导数

这两个公式的关键区别在于:
- 对于输出层神经元,误差项 $\delta_j$ 直接由误差 $E$ 对输出 $o_j$ 的偏导数和激活函数的导数计算得到
- 对于隐藏层神经元,误差项 $\delta_j$ 通过下一层(下游)神经元的误差项 $\delta_k$ 和对应权重 $w_{kj}$ 的加权和,再乘以激活函数的导数计算得到

通过应用这些公式,可以从输出层开始,逐层反向传播误差,计算每个神经元的误差项 $\delta$。然后,使用这些误差项计算权重的梯度,并更新权重以最小化误差。

### Softmax-with-Loss Layer

![softmax & loss](./深度学习入门.assets/3.png)

## 神经网络的训练

- 最优权重参数的最优化方法
  - SGD with Momentum
  - AdaGrad
  - Adam
- 权重参数的初始值
- 超参数的设定方法
- 过拟合，正则化方法
  - 权重值衰减
  - Dropout
- Batch Normalization
- Layer Normalization

### SGD 优化器

SDG 的缺陷：

- 如果函数的形状非均向(anisotropic)，搜索路径会非常低效
- 为了解决这个问题，引入动量 Momentum

### SGD with Momentum

$$
v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)
$$

$$
\theta = \theta - v_t
$$

其中:
- $v_t$ 表示时间步 $t$ 的速度(velocity)向量
- $\gamma$ 表示动量(momentum)系数,通常取值在 0 到 1 之间
- $v_{t-1}$ 表示上一时间步 $t-1$ 的速度向量
- $\eta$ 表示学习率
- $\nabla_\theta J(\theta)$ 表示目标函数 $J(\theta)$ 对参数 $\theta$ 的梯度
- $\theta$ 表示模型的参数向量

SGD with Momentum 的更新过程分为两个步骤:

1. 更新速度向量 $v_t$:
   - 首先,计算当前时间步的梯度 $\nabla_\theta J(\theta)$
   - 然后,将上一时间步的速度向量 $v_{t-1}$ 乘以动量系数 $\gamma$,表示保留一部分上一时间步的速度
   - 再将当前时间步的梯度 $\nabla_\theta J(\theta)$ 乘以学习率 $\eta$,表示根据当前梯度的方向和大小调整速度
   - 最后,将这两项相加,得到更新后的速度向量 $v_t$

2. 更新参数向量 $\theta$:
   - 将当前的参数向量 $\theta$ 减去更新后的速度向量 $v_t$,得到更新后的参数向量 $\theta$

通过引入速度向量 $v_t$ 和动量系数 $\gamma$,SGD with Momentum 可以在一定程度上缓解 SGD 的振荡问题,加速收敛。
- 当连续几次梯度的方向一致时,速度向量会不断累积,加速沿着一致方向的更新
- 当连续几次梯度的方向发生变化时,速度向量会受到抑制,减缓振荡

总之,SGD with Momentum 通过引入速度向量和动量系数,在 SGD 的基础上实现了更平稳、更快速的收敛。

### AdaGrad

AdaGrad 会记录过去所有梯度的平方和。因此，学习越深入，更新的幅度就越小

$$
h \leftarrow h + \frac{\partial L}{\partial W} \odot \frac{\partial L}{\partial W}
$$

$$
W \leftarrow W - \eta \frac{1}{\sqrt{h}} \frac{\partial L}{\partial W}
$$

其中:
- $h$ 表示历史梯度平方和
- $\frac{\partial L}{\partial W}$ 表示损失函数 $L$ 对权重 $W$ 的梯度
- $\odot$ 表示按元素相乘(Hadamard 乘积)
- $\eta$ 表示学习率
- $\sqrt{h}$ 表示对 $h$ 中的每个元素计算平方根

### Adam

Paper: [here](https://arxiv.org/abs/1412.6980)

三个超参数:  
- lr
- $\beta_1$: 0.9
- $\beta_2$: 0.999

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta J(\theta)
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta J(\theta))^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

其中:
- $m_t$ 表示时间步 $t$ 的一阶矩(即梯度的指数加权平均值)
- $v_t$ 表示时间步 $t$ 的二阶矩(即梯度平方的指数加权平均值)
- $\beta_1$ 和 $\beta_2$ 分别表示一阶矩和二阶矩的指数衰减率,通常取值为 0.9 和 0.999
- $\nabla_\theta J(\theta)$ 表示目标函数 $J(\theta)$ 对参数 $\theta$ 的梯度
- $\hat{m}_t$ 和 $\hat{v}_t$ 分别表示校正后的一阶矩和二阶矩估计值
- $\eta$ 表示学习率
- $\epsilon$ 表示一个很小的常数,用于防止分母为零,通常取值为 $10^{-8}$
- $\theta_t$ 表示时间步 $t$ 的参数向量

Adam 的更新过程分为以下步骤:

1. 计算当前时间步的梯度 $\nabla_\theta J(\theta)$

2. 更新一阶矩估计值 $m_t$ 和二阶矩估计值 $v_t$:
   - 将上一时间步的一阶矩估计值 $m_{t-1}$ 乘以衰减率 $\beta_1$,再加上当前梯度乘以 $(1 - \beta_1)$
   - 将上一时间步的二阶矩估计值 $v_{t-1}$ 乘以衰减率 $\beta_2$,再加上当前梯度平方乘以 $(1 - \beta_2)$

3. 计算校正后的一阶矩估计值 $\hat{m}_t$ 和二阶矩估计值 $\hat{v}_t$:
   - 将一阶矩估计值 $m_t$ 除以 $(1 - \beta_1^t)$
   - 将二阶矩估计值 $v_t$ 除以 $(1 - \beta_2^t)$

4. 更新参数向量 $\theta$:
   - 将当前参数向量 $\theta_t$ 减去学习率 $\eta$ 乘以校正后的一阶矩估计值 $\hat{m}_t$ 除以校正后的二阶矩估计值的平方根加上 $\epsilon$

Adam 通过结合 AdaGrad 和 RMSprop 的优点,自适应地调整每个参数的学习率。它利用一阶矩估计值和二阶矩估计值来适应不同的梯度大小和方差,从而实现快速、稳定的收敛。

### 权重的初始化

为什么要对网络的权重初始化，先来看隐藏层的激活值的分布:  

(1) 实验是，向一个5层神经网络(激活函数使用 sigmoid 函数)传入随机生成的输入数据，用直方图绘制各层激活值的数据分布

- Sigmoid 函数，随着输出靠近0或1，导数的值逐渐接近0
- 因此下图的激活值分布会导致梯度消失(gradient vanishing)的问题，并且随着网络的加深，这种现象会更严重

![标准差为1的高斯分布作为权重初始值时的各层激活值的分布](./深度学习入门.assets/4.png)

- 虽然不会发生梯度消失的问题，但是分布过于有偏向，导致神经元输出基本都相似，表现力不够

![标准差为0.01的高斯分布作为权重初始值时的各层激活值的分布](./深度学习入门.assets/5.png)

- 使用 Xavier 初始值([Paper here](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)) 初始化网络的权重
- Xavier 初始化已经成为深度学习中非常常用的权重初始化方法之一,特别是在使用 Sigmoid 和 Tanh 激活函数的网络中

![Xavier 初始值](./深度学习入门.assets/6.png)

(2) 当激活函数使用 ReLU 时，一般推荐使用 ReLU 专用的初始值，也就是Kaiming He等人推荐的初始值，也称为“He初始值”

当前一层的节点数为 n 时，He 初始值使用标准差为 $\sqrt{\frac{2}{n}}$ 的高斯分布。当 Xavier 初始值是 $\sqrt{\frac{1}{n}}$ 时，(直观上)可以解释为，因为 ReLU 的负值区域的值为 0，为了使它更有广度，所以需要 2 倍的系数。

![Why He init_val in relu](./深度学习入门.assets/7.png)

- 使用 std=0.01 时，梯度爆炸，无法学习
- 使用 Xavier 初始值时，网络加深，偏向加深
- 使用 He 初始值时，数据广度一直维持不变，反向传播时也会传递合适的值

**最佳实践:** 当激活函数使用 ReLU 时，权重初始值使用 He 初始值，当激活函数为 sigmoid 或 tanh 等 S 型曲线函数时，初始值使用 Xavier 初始值

### Batch Normalization

为了使各层拥有适当的广度，“强制性”地调整激活值的分布(标准化)，[Paper here](https://arxiv.org/abs/1502.03167)

- 可以使学习快速进行(可以增大学习率)
- 不那么依赖初始值(对于初始值不用那么神经质)
- 抑制过拟合(降低Dropout等的必要性)

将 mini-batch 的输入数据 $\{x_1, x_2, \cdots, x_m\}$ 变换为均值为 0、方差为 1 的数据 $\{\hat{x}_1, \hat{x}_2, \cdots, \hat{x}_m\}$。通过将这个处理插入到激活函数的前面(或者后面)A，可以减小数据分布的偏向。

$$
\begin{aligned}
\mu_B & \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_i \\
\sigma_B^2 & \leftarrow \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2 \\
\hat{x}_i & \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
\end{aligned}
$$

接着，Batch Norm 层会对标准化后的数据进行缩放和平移。

$$
y_i \leftarrow \gamma \hat{x}_i + \beta \qquad (6.8)
$$

这里, $\gamma$ 和 $\beta$ 是参数。一开始 $\gamma = 1$, $\beta = 0$, 然后再通过学习调整到合适的值。

Batch Norm 的计算图和反向传播, [Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)

![Batch Normalization的计算图](./深度学习入门.assets/8.png)

### 过拟合和正则化

过拟合主要的成因：

- 模型拥有大量参数、表现力强
- 训练数据少

L2 正则化: 如果将权重记为 $W$，L2 范数的权值衰减就是 ，然后将这个 $\frac{1}{2}\lambda W^2$ 加到损失函数上。

Dropout: 如果网络的模型变得很复杂，还可以加用 Dropout 层；Dropout 是一种在学习的过程中随机删除神经元的方法。  
- 训练时，每传递一次数据，就会随机选择要删除的神经元
- 测试时，虽然会传递所有的神经元信号，但是对于各个神经元的输出，要乘上训练时的删除比例后再输出
- 因为训练时如果进行恰当的计算的话，正向传播时单纯地传递数据就可以了(不用乘以删除比例)，所以深度学习的框架中进行了这样的实现

### 验证集和超参数

- 常见超参数: 各层神经元数量、batch大小、LR、正则项
- 训练数据用于参数(权重和偏置)的学习
- 验证数据用于调整超参数
- 测试数据用于评估性能和泛化能力

在进行神经网络的超参数的最优化时，与网格搜索等有规律的搜索相比，随机采样的搜索方式效果更好。这是因为在多个超参数中，各个超参数对最终的识别精度的影响程度不同。

超参数最优化的方法：  
- 实践性方式：不断实验缩小超参数的范围
- 更加严密、高效地进行最优化方式：贝叶斯最优化(Bayesian optimization), [here](https://arxiv.org/abs/1206.2944)

## CNN

### 基本概念

- 填充 padding
- 步幅 stride

$$
\begin{aligned}
O &= \left\lfloor\frac{I - K + 2P}{S}\right\rfloor + 1 \\
I &= (O - 1) \times S + K - 2P
\end{aligned}
$$

其中:

- $O$ 表示输出维度(Output dimension)
- $I$ 表示输入维度(Input dimension)
- $K$ 表示卷积核大小(Kernel size)
- $P$ 表示填充量(Padding)
- $S$ 表示步幅(Stride)
- $\lfloor \cdot \rfloor$ 表示向下取整操作

### 卷积操作

单条数据卷积操作：

![单条数据输入和多个卷积核](./深度学习入门.assets/9.png)

批处理卷积操作：

![批处理输入和多个卷积核](./深度学习入门.assets/10.png)

### 池化层

一般来说，池化的窗口大小会和步幅设定成相同的值。

类别:  
- Min Max 池化
- Average 池化

特点:  
- 没有要学习的参数
- 通道数不会有变化
- 对微小的位置变化有鲁棒性(池化会吸收输入数据的偏差)

### 卷积层的实现

在 CNN 卷积层的实现中使用 im2col 函数的主要目的是为了将输入图像或特征图转换为适合矩阵乘法的格式,从而加速卷积操作的计算。

卷积操作本质上是对输入图像的局部区域进行滑动窗口操作,并与卷积核进行逐元素乘法和求和。然而,这种直接的实现方式计算效率较低,尤其是在处理大规模数据时。

im2col 函数的作用是将输入图像或特征图的局部区域展开为列向量,并将这些列向量按顺序排列形成一个矩阵。同时,卷积核也被展开为一个行向量。通过这种转换,卷积操作可以转化为矩阵乘法操作。

优点:  
- 加速计算
- 便于并行化
- 简化实现、兼容性

forward 实现:

![卷积操作细节](./深度学习入门.assets/11.png)

backward 实现:

需要实现 im2col 函数的逆处理，col2im 函数将梯度矩阵转换回与输入图像或特征图相同的形状。col2im 函数的作用与 im2col 函数相反,它将展开的梯度矩阵重新组合成与输入图像或特征图相同的形状,以便继续进行反向传播。

col2im 的作用:

- 梯度还原
- 保持空间信息

### 池化层的实现

forward 实现:

1. 展开输入数据。
2. 求各行的最大值(或最小值、平均值)。
3. 转换为合适的输出大小。

![最大池化层的实现流程](./深度学习入门.assets/12.png)

### CNN 可视化

![CNN学习和可视化](./深度学习入门.assets/13.png)

右边的有规律的滤波器在“观察”什么: 在观察边缘(颜色变化的分界线)和斑块(局部的块状区域)等

AlexNet 各个卷积的注意的是什么: 如果堆叠了多层卷积层，则随着层次加深，提取的信息也愈加复杂、抽象

![AlexNet](./深度学习入门.assets/14.png)

### 具有代表性的 CNN

- LeNet, [here](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

![LeNet](./深度学习入门.assets/15.png)

- AlexNet, [here](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

![AlexNet](./深度学习入门.assets/16.png)

## Deep Learning

如何实现一个性能很好的模型:

- 为神经网络增加复杂度
- 集成学习
- 正则化(学习率衰减, Dropout)
- 数据增强(Data Augmentation)
  - 变形
  - 裁剪
  - 掩码

为什么要加深网络:

- 关于加深层的重要性，现状是理论研究还不够透彻
- 第一个好处可以减少网络的参数数量: 就是与没有加深层的网络相比，加深了层的网络可以用更少的参数达到同等水平(或者更强)的表现力
- 叠加小型滤波器来加深网络的好处是可以减少参数的数量，扩大感受野(receptive field，给神经元施加变化的某个局部空间区域)
- 另一个好处就是使学习更加高效: 通过加深网络，就可以分层次地分解需要学习的问题，传递信息

## ILSVRC - ImageNet

- AlexNet: LeNet 的引申
- VGG: 增加了深度
- GoogLeNet: 不仅深度还增加宽度，引入 Inception 结构

![GoogLeNet Inception](./深度学习入门.assets/17.png)

- ResNet: 进一步加深，并且为了解决过深网络很难学习的问题(梯度消失)，提出残差连接 $F(x)=F(x)+x$ 使得反向传播时信号可以无衰减地传递

![ResNet Redisual Block](./深度学习入门.assets/18.png)

![ResNet Structure](./深度学习入门.assets/19.png)

## 迁移学习

迁移学习: 将学习完的权重(的一部分)复制到其他神经网络，进行再微调(fine tuning)。

## 深度学习的资源管理

- GPU
- 分布式
- 运算精度的位数缩减

## 深度学习的应用案例

- 物体检测
- 图像分割
- 图像标题生成
