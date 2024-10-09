## Model

| 年份 | 名字                                                         | 简介                                                         | 精读                              |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------- |
| 2017 | [Transformer](https://arxiv.org/abs/1706.03762)              | 继 MLP、CNN、RNN 后的第四大类架构                            | [here](./research/transformer.md) |
| 2018 | [ELMo](https://arxiv.org/abs/1802.05365)                     | 使用预训练的双向语言模型(biLM)的内部状态来学习词向量         | here                              |
| 2018 | [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | 使用 Transformer 解码器来做预训练                            | here                              |
| 2018 | [BERT](https://arxiv.org/abs/1810.04805)                     | 使用 Transformer 编码器来做预训练，Transformer 一统 NLP 的开始 | [here](./research/bert.md)        |
| 2019 | [T5](https://arxiv.org/pdf/1910.10683)                       | 使用 Transformer 解码器和编码器，文本到文本格式的预训练      |                                   |
| 2019 | [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 更大的 GPT 模型，朝着 zero-shot learning 迈了一大步          | here                              |
| 2020 | [GPT-3](https://arxiv.org/abs/2005.14165)                    | 100 倍更大的 GPT-2，few-shot learning 效果显著               | here                              |
| 2022 | [InstructGPT](https://arxiv.org/abs/2203.02155)              | 使用人类反馈对GPT-3进行指令微调                              |                                   |
| 2022 | [ChatGPT](https://openai.com/index/chatgpt/)                 | 基于InstructGPT的对话式AI助手                                |                                   |
| 2023 | [GPT-4](https://cdn.openai.com/papers/gpt-4.pdf)             | We used python😂 多模态大模型，支持图片和文本的输入，文本的输出 | here                              |
| 2023 | [Llama](https://arxiv.org/abs/2302.13971)                    | Meta开源LLM向闭源大模型发出冲锋号角，参数量7B到65B           |                                   |
| 2023 | [Llama 2](https://arxiv.org/abs/2307.09288)                  | 70B到70B参数量开源大模型可能成为闭源大模型替代品             |                                   |
| 2024 | [Llama 3.1](https://arxiv.org/pdf/2407.21783)                | 强大的 Meta 开源模型 - 动态扩展，多模态学习，零样本学习，高效计算 |                                   |

## Literature Review

| 年份 | 名字                                                         | 简介                                                   | 精读 |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------ | ---- |
| 2020 | [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) | Transformer 模型的全面综述                             | here |
| 2022 | `*` [On the Opportunities and Risks of Foundation Models](https://arxiv.org/pdf/2108.07258) | LLM 必看综述: LLM 能力、机遇和挑战，垂直领域应用，影响 |      |
| 2023 | `*` [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) | LLM 必看综述: 资源、预训练、微调、应用、能力等         | here |
| 2023 | [Summary of ChatGPT-Related Research and Perspective Towards the Future of Large Language Models](https://arxiv.org/abs/2304.01852) | 对 ChatGPT 相关研究进行了全面综述                      |      |
| 2023 | [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) | 全面的大模型评测综述                                   |      |
| 2024 | [What is the Role of Small Models in the LLM Era: A Survey](https://arxiv.org/abs/2409.06857) | 从协作和竞争关系来看 LLM 和 SM                         |      |
| 2024 | [A Survey of Large Language Models for Graphs](https://arxiv.org/abs/2405.08011) | 综述不同设计方法来整合 LLMs 和图学习技术               |      |

## Optimization of Model Architecture

| 年份 | 名字                                                                                                                         | 简介                                                                                        | 精读 |
| ---- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ---- |
| 2019 | [Understanding and Improving Layer Normalization](https://arxiv.org/abs/1911.07013)                                          | 深入分析层归一化机制，发现其效果主要源于均值和方差的导数而非前向归一化                      ||
| 2021 | [Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth](https://arxiv.org/abs/2103.03404) | 纯注意力模型的输出随深度呈双指数级退化为秩为1的矩阵，而跳跃连接和多层感知器可以防止这种退化 |      |
| 2022 | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) | FlashAttention是一种新的注意力计算方法，通过IO感知设计显著提高了Transformer模型在长序列处理时的计算效率和内存使用 | |
| 2023 | [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) | 通过改进并行化和工作分区策略来进一步提高注意力机制的计算效率 | |
| 2024 | [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) | 过引入异步计算和低精度运算来进一步提高注意力机制的速度和准确性 | |

## Quantization

| 年份 | 名字                                                         | 简介                                                         | 精读 |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| 2024 | [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764) | 微软研究院提出的一个模型架构，采用极端量子量化，仅用-1, 0, 1表示每个参数，每个参数仅使用 $log_2(3)=1.58$ 比特 |      |

## Fine-Tuning

| 年份 | 名字                                                         | 简介                               | 精读 |
| ---- | ------------------------------------------------------------ | ---------------------------------- | ---- |
| 2021 | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | PERF参数化高效微调: Low rank, LoRA |      |

## RAG

| 年份 | 名字                                                         | 简介                                                         | 精读 |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| 2020 | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | RAG 开山之作: 结合预训练语言模型和外部知识检索，解决知识密集型任务 | here |
| 2023 | [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) | 对RAG在LLM中的应用进行全面综述                               |      |
| 2024 | [Retrieval Augmented Generation (RAG) and Beyond: A Comprehensive Survey on How to Make your LLMs use External Data More Wisely](https://arxiv.org/abs/2409.14924) | 微软出品综述: 详细探讨了使用更有效的技术将外部数据集成到LLMs |      |

## Reinforcement Learning

| 年份 | 名字                                                         | 简介                                                         | 精读 |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| 2017 | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | 强化学习必读算法，通过环境交互采样和优化替代目标函数，实现多轮小批量更新 |      |
| 2020 | [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) | 开辟进行基于人类反馈的强化模型训练，通过摘要任务展示         |      |
| 2023 | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | InstructGPT: 通过人类反馈来微调语言模型                      |      |

## Chain of Thought

| 年份 | 名字                                                                                                          | 简介                                                                                                                                                                               | 精读 |
| ---- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| 2023 | [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)     | 探索思维链提示技术(CoT)如何显著提高大型语言模型的复杂推理能力                                                                                                                      |      |
| 2023 | [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)                             | 大型语言模型具有强大的零样本推理能力和潜力                                                                                                                                         |      |
| 2023 | [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)   | 提出了一种名为"自一致性"的新解码策略，通过采样多条推理路径并选择最一致的答案                                                                                                       |      |
| 2024 | [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) | 开辟新的语言模型训练方法，通过让模型在生成每个 token 时学习产生解释性的内部思考，提高了模型在预测困难 token 和回答复杂问题时的能力，无需针对特定任务进行微调就能实现零样本性能提升 |      |

## Code

| 年份 | 名字                                                         | 简介                                               | 精读 |
| ---- | ------------------------------------------------------------ | -------------------------------------------------- | ---- |
| 2021 | [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) | 公开代码微调的编程大模型，有局限也有未来发展       |      |
| 2022 | [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814) | 展示解决高难度编程问题的潜力，能生成竞赛级别的代码 |      |
| 2024 | [Qwen2.5-Coder Technical Report](https://huggingface.co/papers/2409.12186) | 阿里同义千问 Qwen2.5-Coder 技术报告                |      |

## RWKV

RWKV (Receptance Weighted Key Value)

| 年份 | 名字 | 简介 | 精读 |
| ---- | ---- | ---- | ---- |
|      |      |      |      |

## Mamba

| 年份 | 名字 | 简介 | 精读 |
| ---- | ---- | ---- | ---- |
|      |      |      |      |
