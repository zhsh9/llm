# Tensor

## Initialize

| 初始化类别              | 方法                                                         | 描述                                   | 常见用途                                   |
| ----------------------- | ------------------------------------------------------------ | -------------------------------------- | ------------------------------------------ |
| 从Python列表或数组创建  | `torch.tensor()`, `torch.from_numpy()`                       | 从Python列表或NumPy数组创建tensor      | 将已有数据转换为PyTorch tensor             |
| 使用PyTorch内置函数创建 | `torch.zeros()`, `torch.ones()`, `torch.arange()`, `torch.linspace()`, `torch.eye()` | 创建具有特定模式或值的tensor           | 初始化特定结构的数据，如全零矩阵或单位矩阵 |
| 随机初始化              | `torch.rand()`, `torch.randn()`, `torch.randint()`           | 创建包含随机值的tensor                 | 神经网络权重初始化，生成随机数据           |
| 指定数据类型和设备      | `tensor(..., dtype=torch.float32)`, `tensor(..., device='cuda:0')` | 创建特定数据类型或在特定设备上的tensor | 控制数值精度，利用GPU加速计算              |
| 改变tensor形状          | `.reshape()`, `.view()`                                      | 改变现有tensor的形状                   | 调整数据结构以适应模型输入或进行特定操作   |
| 类似现有tensor的初始化  | `torch.zeros_like()`, `torch.ones_like()`, `torch.rand_like()` | 创建与给定tensor形状相同的新tensor     | 创建与现有数据结构匹配的新tensor           |
| 通过概率分布初始化      | `torch.bernoulli()`, `torch.normal()`, `torch.multinomial()`, `torch.poisson()`, `torch.rand()` | 从特定概率分布中采样创建tensor         | 生成符合特定统计分布的数据，模拟随机过程   |

## Attributes

| 属性          | 类型                 | 描述                             | 示例                                | 常见用途                           |
| ------------- | -------------------- | -------------------------------- | ----------------------------------- | ---------------------------------- |
| shape         | torch.Size           | 返回tensor的形状（各维度的大小） | `x.shape` → `torch.Size([3, 4, 5])` | 检查tensor的结构，调整数据形状     |
| dtype         | torch.dtype          | 返回tensor的数据类型             | `x.dtype` → `torch.float32`         | 确保数据类型正确，进行类型转换     |
| device        | torch.device         | 返回tensor所在的设备（CPU或GPU） | `x.device` → `cuda:0`               | 确定计算设备，在CPU和GPU间移动数据 |
| requires_grad | bool                 | 指示是否需要为该tensor计算梯度   | `x.requires_grad` → `True`          | 控制自动微分，设置模型参数         |
| is_leaf       | bool                 | 指示该tensor是否为叶子节点       | `x.is_leaf` → `True`                | 理解计算图结构，调试反向传播       |
| grad          | torch.Tensor 或 None | 存储tensor的梯度（如果已计算）   | `x.grad` → `tensor([1., 1., 1.])`   | 访问计算得到的梯度，进行梯度分析   |
| data          | torch.Tensor         | 返回tensor的数据，不会被追踪梯度 | `x.data` → `tensor([1., 2., 3.])`   | 访问底层数据，不触发自动微分       |
| numel()       | int                  | 返回tensor中元素的总数           | `x.numel()` → `60`                  | 计算tensor大小，内存使用分析       |

## Data Types

[torch.Tensor #Data Types — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/tensors.html#data-types)

| 分类   | 数据类型         | 别名         | 位数 | 范围/精度         | 典型用途                             |
| ------ | ---------------- | ------------ | ---- | ----------------- | ------------------------------------ |
| 浮点型 | torch.float32    | torch.float  | 32   | 单精度            | 默认类型，常用于大多数计算           |
|        | torch.float64    | torch.double | 64   | 双精度            | 需要高精度计算                       |
|        | torch.float16    | torch.half   | 16   | 半精度            | 减少内存使用，GPU训练                |
|        | torch.bfloat16   | -            | 16   | 脑浮点            | 专为神经网络设计，某些硬件上性能更好 |
| 整型   | torch.int8       | -            | 8    | -128 to 127       | 量化模型，节省内存                   |
|        | torch.uint8      | -            | 8    | 0 to 255          | 图像处理，存储小的正整数             |
|        | torch.int16      | torch.short  | 16   | -32768 to 32767   | 中等范围的整数计算                   |
|        | torch.int32      | torch.int    | 32   | -2^31 to 2^31-1   | 大范围的整数计算                     |
|        | torch.int64      | torch.long   | 64   | -2^63 to 2^63-1   | 非常大的整数，如索引                 |
| 布尔型 | torch.bool       | -            | 1    | True/False        | 掩码操作，条件语句                   |
| 复数型 | torch.complex64  | -            | 64   | 32位实部+32位虚部 | 信号处理，傅里叶变换                 |
|        | torch.complex128 | -            | 128  | 64位实部+64位虚部 | 高精度复数计算                       |

数据类型标准：

- Float: IEEE 754, 32bit = 1 sign + 8 exponent + 23 fraction


## Operations

[torch #Tensors — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/torch.html#tensors)

- Creating Ops
- Indexing, Slicing, Joining, Mutating Ops

| 操作类别 | 主要函数/方法 | 描述 | 常见用途 |
|----------|---------------|------|----------|
| 算术运算 | `+`, `-`, `*`, `/`, `torch.add()`, `torch.sub()`, `torch.mul()`, `torch.div()`, `torch.matmul()` | 执行基本的算术运算和矩阵乘法 | 数值计算、特征组合、层间计算 |
| 统计运算 | `torch.mean()`, `torch.sum()`, `torch.max()`, `torch.min()` | 计算tensor的统计量 | 数据分析、损失函数计算、特征归一化 |
| 形状操作 | `reshape()`, `view()`, `t()`, `transpose()`, `unsqueeze()`, `squeeze()` | 改变tensor的形状或维度 | 数据预处理、维度匹配、批处理 |
| 索引和切片 | `tensor[index]`, `tensor[start:end]` | 访问或修改tensor的特定元素或子集 | 数据选择、特征提取、掩码操作 |
| 拼接和分割 | `torch.cat()`, `torch.stack()`, `torch.chunk()` | 合并多个tensors或将一个tensor分割成多个 | 数据组合、批处理、模型集成 |
| 数学函数 | `torch.sin()`, `torch.cos()`, `torch.exp()`, `torch.log()` | 执行各种数学函数运算 | 特征变换、激活函数、概率计算 |
| 比较操作 | `torch.eq()`, `torch.gt()`, `torch.lt()`, `torch.logical_and()`, `torch.logical_or()` | 执行元素间的比较和逻辑运算 | 条件筛选、掩码生成、逻辑控制 |
| 条件和逻辑操作 | `torch.where()`, `torch.masked_select()`, `torch.nonzero()` | 基于条件选择元素或执行复杂的逻辑操作 | 条件替换、掩码应用、稀疏操作 |

## Bridge with NumPy

| 操作类别                      | 主要函数/方法              | 描述                                                | 注意事项                              | 常见用途                                 |
| ----------------------------- | -------------------------- | --------------------------------------------------- | ------------------------------------- | ---------------------------------------- |
| NumPy Array 转 PyTorch Tensor | `torch.from_numpy()`       | 将NumPy array转换为PyTorch tensor                   | 共享内存，修改一个会影响另一个        | 将预处理后的NumPy数据导入PyTorch模型     |
| PyTorch Tensor 转 NumPy Array | `tensor.numpy()`           | 将PyTorch tensor转换为NumPy array                   | 仅适用于CPU上的tensor                 | 将模型输出转换为NumPy进行后续处理        |
| GPU Tensor 转 NumPy Array     | `gpu_tensor.cpu().numpy()` | 将GPU上的tensor转换为NumPy array                    | 需要先将tensor移到CPU                 | 将GPU计算结果转换为NumPy进行可视化或保存 |
| 保持数据类型一致性            | `torch.from_numpy()`       | 在转换过程中保持数据类型不变                        | 注意float64和float32的区别            | 确保数值精度在转换过程中不丢失           |
| 处理不同的内存布局            | `tensor.contiguous()`      | 处理非连续内存的tensor                              | 可能涉及内存复制，影响性能            | 确保tensor在某些操作中的兼容性           |
| 共享内存和数据同步            | `torch.from_numpy()`       | NumPy array和PyTorch tensor共享内存                 | 修改一个会影响另一个                  | 在NumPy和PyTorch间高效地共享大型数据集   |
| 处理复杂数据类型              | `torch.from_numpy()`       | 转换复数等特殊数据类型                              | 确保PyTorch支持相应的数据类型         | 处理信号处理或科学计算中的复数数据       |
| 共享内存检查                  | `tensor.is_contiguous()`   | 检查tensor是否在内存中连续存储                      | 连续存储的tensor更可能与NumPy共享内存 | 优化内存使用和性能                       |
| 强制复制                      | `tensor.clone().detach()`  | 创建tensor的独立副本                                | 用于避免共享内存时的意外修改          | 在需要独立副本的场景中使用               |
| 性能优化                      | -                          | 尽可能使用 `torch.from_numpy()` 和 `tensor.numpy()` | 这些操作通常不会复制数据，效率更高    | 提高大规模数据处理的效率                 |
