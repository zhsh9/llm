# [BERT](https://arxiv.org/abs/1810.04805): Bidirectional Encoder Representations from Transformers

Deep bidirectional Transformers (Encoder-only, MLM) + Next sentence prediction = Token-level + Sentence-level

Contributions:

- bidirectional pre-training
- reduce task-specific architecture = unify architecture accross different tasks
- sota performance over nlp tasks
- [google-research/bert: TensorFlow code and pre-trained models for BERT (github.com)](https://github.com/google-research/bert)

Related work

- Unsupervised Feature-based Approaches
- Unsupervised Fine-tuning Approaches
- Transfer Learning from Supervised Data
- [The Annotated Transformer (harvard.edu)](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

BERT = pre-training (unlabeled) + fine-tuning (labeled)

- Architecture: multi-layer bidirectional Transformer encoders
- Implementation: tensor2tensor library

| Variable | Meaning                               |
| -------- | ------------------------------------- |
| L        | Number of layers (Transformer blocks) |
| H        | Hidden size                           |
| A        | Number of self-attention heads        |

| Model      | L (Layers) | H (Hidden size) | A (Attention heads) | Total Parameters |
| ---------- | ---------- | --------------- | ------------------- | ---------------- |
| BERT_BASE  | 12         | 768             | 12                  | 110M             |
| BERT_LARGE | 24         | 1024            | 16                  | 340M             |
