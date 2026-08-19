# Mini GPT from Scratch

A small **decoder-only language model built from scratch in PyTorch**, developed through a series of experiments — starting with TinyStories and gradually moving toward OpenOrca-based pretraining and conversational text generation.

This repository documents the complete journey: from implementing the Transformer architecture and training pipeline to debugging GPU limitations and building a small conversational language model.

> **This is an experimental Small Language Model (SLM), not a pretrained foundation model.**

---

## 🚀 Project Journey

The project evolved through three major stages:

```text
TinyStories
    │
    ▼
01 · Foundation SLM
    │
    │  Learn the complete training pipeline
    ▼
02 · OpenOrca Pretraining
    │
    │  Larger + more diverse instruction-style data
    ▼
03 · Conversational SLM
    │
    │  Conversational + identity data
    ▼
Small GPT-style Conversational Model
```

A failed intermediate experiment is also preserved separately because it contains useful GPU/CUDA debugging work.

---

## ✨ What I Built

The model is a GPT-style **decoder-only Transformer** implemented directly in PyTorch.

The implementation includes:

* Token and positional embeddings
* Causal self-attention
* Multi-head attention
* Transformer blocks
* Layer normalization
* GELU activation
* Residual connections
* Weight tying
* Causal language-modeling objective
* Autoregressive text generation
* Temperature sampling
* Top-k sampling
* Gradient accumulation
* Gradient clipping
* AdamW optimization
* Learning-rate warmup
* Cosine learning-rate decay
* Mixed-precision training
* Training/validation loss monitoring
* Model checkpointing

The model is trained from **randomly initialized weights** rather than starting from a pretrained GPT checkpoint.

---

## 🧠 Model Architecture

The core architecture follows the GPT/decoder-only Transformer design.

### Configuration

| Parameter           |                           Value |
| ------------------- | ------------------------------: |
| Architecture        |        Decoder-only Transformer |
| Layers              |                               6 |
| Attention heads     |                               6 |
| Embedding dimension |                             384 |
| Vocabulary size     |                          50,257 |
| Context length      | 128–256 depending on experiment |
| Activation          |                            GELU |
| Normalization       |                       LayerNorm |
| Optimizer           |                           AdamW |
| Precision           |      FP16 / BF16 when supported |

The largest conversational experiment contains approximately **30M trainable parameters**.

---

## 📚 Experiments

### 01 — TinyStories Foundation

`notebooks/01_tinystories_foundation.ipynb`

The first working stage of the project.

The model is trained on the **TinyStories** dataset and establishes the complete language-modeling pipeline:

```text
Dataset
  ↓
GPT-2 Tokenization
  ↓
Tokenized binary data
  ↓
Batch generation
  ↓
Transformer
  ↓
Language-model loss
  ↓
Training
  ↓
Autoregressive generation
```

This experiment was primarily about understanding how a GPT-style language model works internally.

---

### 02 — OpenOrca Pretraining

`notebooks/02_openorca_pretraining.ipynb`

The next stage moved beyond TinyStories toward a larger instruction-oriented dataset.

The experiment involved:

* Dataset filtering
* Tokenization
* Binary dataset preparation
* Larger-scale training
* Validation loss monitoring
* Checkpointing
* Text generation

The model was scaled to approximately **30M parameters**.

This stage demonstrated how changing the training data and scale affects the behavior of a small language model.

---

### 03 — Conversational SLM

`notebooks/03_conversational_slm.ipynb`

The final major experiment focuses on conversational behavior.

The training data combines conversational/instruction-style examples with identity-related examples so that the model can produce more assistant-like responses.

### Reported configuration

* ~30M parameters
* 6 Transformer layers
* 6 attention heads
* 384-dimensional embeddings
* GPT-2 tokenizer
* 50,257-token vocabulary
* 256-token context
* 15,000 training steps
* Best reported validation loss: **2.6913**

The notebook also includes interactive text generation after training.

---

## 🧪 Failed Experiment: CUDA Debugging

`experiments/openorca_cuda_debugging.ipynb`

Not every experiment worked.

An intermediate OpenOrca training attempt encountered CUDA compatibility issues with the available Tesla P100 GPU and the installed PyTorch build.

The experiment is intentionally preserved because it documents part of the actual development process:

```text
Training attempt
      ↓
CUDA compatibility error
      ↓
GPU compute capability mismatch
      ↓
Environment investigation
      ↓
Training configuration changes
      ↓
Later successful experiment
```

Rather than hiding failed experiments, this repository keeps the debugging trail separate from the main successful pipeline.

---

## 🏗️ Repository Structure

```text
mini-gpt-from-scratch/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_tinystories_foundation.ipynb
│   ├── 02_openorca_pretraining.ipynb
│   └── 03_conversational_slm.ipynb
│
└── experiments/
    └── openorca_cuda_debugging.ipynb
```

---

## 🔧 Tech Stack

* **Python**
* **PyTorch**
* **NumPy**
* **Hugging Face Datasets**
* **tiktoken**
* **Matplotlib**
* **CUDA / GPU acceleration**

---

## ⚙️ Training Pipeline

The overall training process is:

```text
Raw Dataset
     ↓
Dataset Cleaning / Filtering
     ↓
GPT-2 Tokenization
     ↓
Binary Token Storage
     ↓
Random Context Sampling
     ↓
Input / Target Sequence Creation
     ↓
Decoder-only Transformer
     ↓
Cross-Entropy Loss
     ↓
AdamW
     ↓
Gradient Accumulation
     ↓
Learning Rate Scheduling
     ↓
Checkpointing
     ↓
Autoregressive Generation
```

---

## 📈 Training Strategy

The training loop incorporates several techniques commonly used when training Transformer language models:

### Optimizer

AdamW with weight decay is used for optimization and regularization.

### Learning-rate scheduling

Training uses:

1. Linear warmup
2. Cosine decay

### Gradient accumulation

Gradient accumulation is used to obtain a larger effective batch size while working within GPU memory constraints.

### Gradient clipping

Gradients are clipped to improve training stability.

### Mixed precision

FP16/BF16 autocasting is used when supported by the available hardware.

---

## 💬 Generation

After training, the model generates text autoregressively.

Generation supports:

* Temperature control
* Top-k sampling
* Context-window truncation

Example:

```text
Prompt
  ↓
Tokenize
  ↓
Model predicts next token
  ↓
Sample token
  ↓
Append token
  ↓
Repeat
  ↓
Decode generated sequence
```

---

## 📊 Results

The experiments showed progressive improvement in the model's ability to generate structured text.

| Stage | Dataset             | Model                       | Training                |
| ----- | ------------------- | --------------------------- | ----------------------- |
| V1    | TinyStories         | Small GPT-style Transformer | Foundation experiment   |
| V2    | OpenOrca            | ~30M Transformer            | Failed / CUDA debugging |
| V3    | OpenOrca            | ~30M Transformer            | 25K-step experiment     |
| V4    | Conversational data | ~30M Transformer            | 15K-step experiment     |

The best reported validation loss in the conversational experiment was:

**2.6913**

---

## ⚠️ Limitations

This project is intentionally small and experimental.

The model:

* Has only ~30M parameters
* Is trained on relatively limited data compared with modern LLMs
* Does not have the knowledge or reasoning capabilities of large language models
* Has limited context length
* Can generate repetitive or incorrect responses
* Has not undergone instruction alignment comparable to production assistants
* Should not be treated as a reliable knowledge source

The goal is **understanding and implementation**, not competing with large-scale LLMs.

---

## 🎯 Why This Project?

Large language models can feel like black boxes when accessed only through APIs.

This project was built to understand what happens underneath:

> **How does a language model actually learn to predict the next token?**

Instead of starting with a pretrained model, I implemented the core GPT-style architecture myself and trained it through multiple iterations.

The project therefore focuses as much on the **learning journey and engineering process** as on the final model.

---

## 🔬 Future Improvements

Potential next steps include:

* [ ] Extract reusable model code into `src/`
* [ ] Improve dataset quality and diversity
* [ ] Train for longer
* [ ] Experiment with larger context windows
* [ ] Compare different model sizes
* [ ] Add validation metrics beyond loss
* [ ] Add systematic generation benchmarks
* [ ] Experiment with learning-rate schedules
* [ ] Add instruction tuning
* [ ] Add a lightweight inference interface
* [ ] Publish trained checkpoints

---

## 📌 Note on the Tokenizer

The model uses the **GPT-2 tokenizer/vocabulary** rather than implementing a tokenizer from scratch.

The Transformer language model itself is implemented and trained from scratch using PyTorch.

---

## 📜 License

Add your preferred license before publishing the repository.

---

## ⭐ Project Status

**Experimental / Educational**

The repository represents an iterative exploration of training a small GPT-style language model from scratch, including successful experiments, failed experiments, and infrastructure debugging.
