# [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Key Concept

- tokenization
- input embedding
- position encoding
- residual
- q, k, v
- add & layer norm
- encoder & decoder
- attention & self-attention
- multi-head attention
- mask attention
- encoder & decoder attention
- output probability & logit & softmax

## Model Architecture

| 结构                                                                                                                                                                                 | 解析                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="./transformer.assets/image-20240925161556037.png" alt="image-20240925161556037"  />                                                                                        | Inputs, Outputs dim: $d_{model} = 512$<br/>FFN inner-layer dimension: $d_{ff}=2048$<br/>Encoder & Decoder Block Num: $N = 6$<br/>Multi-head Attention's heads: $h = 8$<br/>Q, K, V dimension: $d_k=d_v=d_{model}/h=64$<br />\|V\|: vocabulary size                                                                                                                                                                                                                                                                                    |
| **Embedding Layer (learned):**<br />convert input/output tokens to vectors of dim $d_{model}$                                                                                        | $E(x)=xW_E$<br />$x$ dim: $(L,                                                                                                                                                                                                                                                                                                                                                                                        \| V                                \| )$<br />$W_E$ dim: $( \| V                               \| ,d_{model})$ |
| **Position Encoding:**<br />provides a unique and information-rich representation for each position while maintaining information about the relative relationships between positions | $PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}})$<br />$PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{model}})$                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Encoder & Decoder Stacks**, `sub-layer`:<br />- (Masked) Multi-Head Attention (learned)<br />- Position-wise FNN<br />- Residual Connection & Layer Normalization (learned)        | $\text{LayerNorm}(x+\text{SubLayer}(x))$<br />$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$                                                                                                                                                                                                                                                                                                                                                                                                 |
| <img src="./transformer.assets/image-20240925164820196.png" alt="image-20240925164820196" style="zoom:50%;" />                                                                       | Dot-product attention:<br />$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$<br />Why $\sqrt{d_k}$: To reduce the magnitude of the matrix multiplication result, <br />preventing it from entering regions where the softmax function has very small gradients.                                                                                                                                                                                                                                                                   |
| <img src="./transformer.assets/image-20240925165401027.png" alt="image-20240925165401027" style="zoom:50%;" />                                                                       | <img src="./transformer.assets/image-20240925175605190.png" alt="image-20240925175605190" style="zoom:50%;" />                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Position-wise Feed-Forward Networks (learned)**                                                                                                                                    | $FFN(x)=ReLu(0,xW_1+b_1)W_2+b_2$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Linear (learned) & Softmax**:<br />convert decoder output to predict next-token prob                                                                                               | $z = W_o h_t + b_o$<br />$P(y_i\|x) = \frac{e^{z_i}}{\sum_{j=1}^{ \| V \| } e^{z_j}}$<br />- $W_o$ dim: $ \| V \| \times d_\text{model}$<br />- $h_t$ dim: $d_\text{model}$                                                                                                                                                                                                                                                                                                                                                           |

## Positional Encoding

![image-20240925171448455](./transformer.assets/image-20240925171448455.png)

Patterns:

- The top rows show rapid alternations between positive and negative values, representing high-frequency components.
- As we move down the dimensions, the patterns become more stretched out, showing lower-frequency components.
- The bottom rows show very slow changes, almost appearing as solid stripes.

This visualization demonstrates how positional encoding provides unique patterns for each position while maintaining relative positional information. The varying frequencies across dimensions allow the model to learn both fine-grained and coarse positional relationships.

## Model Analysis

![image-20240926012813684](./transformer.assets/image-20240926012813684.png)

- **Efficient for typical dimensions**: While the complexity appears high, in many practical applications, d (representation dimension) is often much smaller than n (sequence length), making it computationally efficient.
- **Parallel processing**:
  - A major advantage of Self-Attention is its ability to process the entire sequence in parallel.
  - Unlike recurrent layers, it doesn't require sequential processing.
  - This significantly speeds up both training and inference, especially for long sequences.
- **Unrestricted attention span**: Self-Attention allows any position in the sequence to directly attend to any other position, without distance limitations.
- **Flexible dependency modeling**:
  - Self-Attention can easily capture dependencies of various scales and types.
  - It doesn't require predefined fixed window sizes (like convolutions) or fixed time steps (like recurrent layers).
- **Interpretability**: The attention weights in Self-Attention can intuitively explain the model's decision-making process, which is valuable in many applications.

## Experiment

### Training

Data & Batching & Token

- Sentences were encoded using byte-pair encoding
- Sentence pairs were batched together by approximate sequence length

Optimizer

- Adam
- β1 = 0.9, β2 = 0.98 and ε = 10−9
- lr scheduler with warm-up (warmup_steps=4000)

$$
lrate = d_{\text{model}}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
$$

Regulation

- Residual Dropout: $P_{drop}=0.1$
- Label Smoothing: $\epsilon_{ls}=0.1$

Sequence generation by Beam search

- Beam size = 4
- Length penalty $\alpha=0.6$

Max output length = input length + 50 (with early terminating available)

### Result

![image-20240926014003185](./transformer.assets/image-20240926014003185.png)

