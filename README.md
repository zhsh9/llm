# 从 0 开始的 LLM 之旅

Document:

- [Python](https://docs.python.org/3/)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [TensorFlow](https://www.tensorflow.org/api_docs/python/tf)

Application:
- NLP
- CV
- Audio
- RLHF
- LLM

AI System Development:
- Model Dev & Train
  - DeepSpeed
  - Megatron-LM
  - Colossal-AI
  - HuggingFace
- Application Dev & Deploy
  - LangChain
  - Milvus

Training Performance Metrics:
- MFU (Model FLOPs Utilization)
  - MFU = Actual FLOPs / Theoretical Peak FLOPs
- HFU (Hardware FLOPs Utilization)
  - HFU = Actual Hardware FLOPs / Theoretical Peak Hardware FLOPs

Framework:
- Training
  - PyTorch
  - TensorFlow
  - Caffe
  - MindSpore
  - JAX
- Inference
  - MNN
  - ONNX
  - TensorRF

Tokenization Algorithm:
- BPE: GPT-n, RoBERTa, Llama, BART, ChatGLM-6B
- WordPiece: BERT, DistilBERT, MobileBERT
- Unigram: AIBERT, T5, XLNet

Resource of LLM:
- Transformers
- DeepSpeed
- JAX
- BMTrain

Model Structure:
- Encoder only
- Decoder only
- Encoder & Decoder
- MOE

Attention Block:
- Multi-Head Attention (MHA)
- Multi-Head Latent Attention (MLA), KV Cache

Fine-Tuning:
- Supervised FT
- Unsupervised FT
- Self-Supervised FT
- Semi-Supervised FT
- Multi-Task FT
- Adversarial FT
- Continual FT
- Few-Shot FT
- Zero-Shot FT
- Instruction FT

Fine-Tuning Algorithm Focusing on Parameter:
- Full Fine-Tuning
- Parameter-Efficient Fine-Tuning (PEFT)
  - LoRA & AdaLoRA
  - Prefix Tuning
  - Prompt Tuning
- BitFit

Parallel: DP, TP, PP, CP
- DeepSpeed: DP
- Megatron-LM: DP, TP, PP

Fine-Tuning Tool: Perf, Llama Factory

## Roadmap

- [ ] Fluent Python, [here](./Python.md)
- [ ] PyTorch Walk Through, [here](./PyTorch.md)
- [ ] NLP Classic Papers, [here](./LLM%20Paper.md)