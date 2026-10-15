
---
# GPT-2 (124M) from Scratch in PyTorch

A ground-up implementation of the Generative Pre-trained Transformer 2 (GPT-2, 124M parameter variant) base architecture in PyTorch, with custom modules, autoregressive text generation, and weight loading from OpenAI's pre-trained checkpoints.

Developed as part of the Post-Graduate Program in Artificial Intelligence, Department of Informatics, Aristotle University of Thessaloniki (AUTH).

Author: Aristeidis Pagonopoulos

---

### Architectural Overview & Features

This implementation avoids high-level abstractions like Hugging Face `PreTrainedModel` and builds the decoder blocks from foundational tensor operations:

1. **Pre-LayerNorm Architecture (Pre-Norm):**
   - Implemented custom `LayerNorm` (with scale $\gamma$ and shift $\beta$ parameters).
   - Applied normalization before the attention and feed-forward sublayers ($x \leftarrow x + \text{SubLayer}(\text{LayerNorm}(x))$) for stable gradient flow across all 12 layers.
   - Includes modular alternatives (`RMSNorm` and `SwiGLU`) alongside standard components.

2. **Causal Multi-Head Self-Attention:**
   - Multi-head projections for queries, keys, and values ($d_{\text{in}} = 768 \to d_{\text{out}} = 768$, 12 heads, head dimension 64).
   - Scaled dot-product attention with causal upper-triangular masking to enforce autoregressive token conditioning.

3. **Feed-Forward Network (FFN):**
   - $d \to 4d \to d$ projection ($768 \to 3072 \to 768$) with approximate `GELU` activation.

4. **Residual Connections & Dropout:**
   - Dual identity skip connections per block ($x \leftarrow x + \text{Dropout}(\dots)$).

5. **Weight Loading & Text Generation:**
   - Utility to download OpenAI's official 124M GPT-2 checkpoint weights and map them into the custom PyTorch module hierarchy.
   - Autoregressive generation script supporting temperature scaling, top-$k$ filtering, and token decoding via `tiktoken` (BPE).

---

### Parameter Breakdown (GPT-124M)

Configuration: $d = 768$, $n_{\text{heads}} = 12$, $n_{\text{layers}} = 12$, context length $= 1024$, vocab size $= 50,257$.

| Component | Formula | Parameter Count |
| :--- | :--- | :--- |
| **Multi-Head Attention (per block)** | $4d^2$ | 2,359,296 |
| **Feed-Forward Network (per block)** | $8d^2 + 5d$ | 4,722,432 |
| **LayerNorm $\times$ 2 (per block)** | $4d$ | 3,072 |
| **Total per Transformer Block** | | $\approx 7.08\text{ M}$ |
| **All 12 Blocks** | $12 \times 7.08\text{ M}$ | $\approx 84.9\text{ M}$ |
| **Token & Position Embeddings** | $(50,257 \times 768) + (1024 \times 768)$ | $\approx 39.4\text{ M}$ |
| **Total Architecture Budget** | | **$\approx 124.4\text{ M}$** |

---

### Project Structure

- `gpt_custom_model.py`: Model architecture, forward pass, weight-loading functions, and CLI inference entry point.
- `gpt_download.py`: Helper script to download OpenAI GPT-2 checkpoints.
- `presentation.pdf`: Slide deck detailing the mathematical formulation, LayerNorm vs RMSNorm comparisons, and skip-connection dynamics.
- `requirements.txt`: Minimal dependencies.


## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ArisPagonopoulos/gpt2-from-scratch-pytorch.git
cd gpt2-from-scratch-pytorch
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Text with Pre-trained GPT-2 (124M) Weights

```bash
python gpt_custom_model.py "Every journey begins with" \
    --loaded \
    --tokens 30 \
    --temperature 0.8 \
    --top_k 40
```

### 4. Run with Randomly Initialized Weights

```bash
python gpt_custom_model.py "Every journey begins with" --tokens 20
```

---

## Example Output

**Prompt**

```text
Every journey begins with
```

**Generated**ontinuation**

```text
Every journey begins with a single idea, and every idea can evolve
into something much larger when given enough time and attention...
```

*(Output will vary due to stochastic sampling.)*

---

## Model Configuration

The implementation reproduces the GPT-2 Small (124M) architecture:

| Hyperparameter | Value |
|---------------|--------|
| Layers | 12 |
| Attention Heads | 12 |
| Embedding Dimension | 768 |
| FFN Hidden Dimension | 3072 |
| Context Length | 1024 |
| Vocabulary Size | 50,257 |
| Parameters | ~124M |

---

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Tiktoken
- TensorFlow (for official checkpoint loading)

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## Educational Objectives

This project was developed for educational purposes to provide a transparent implementation of:

- Transformer decoder architecture
- Causal self-attention
- Layer normalization
- Residual connections
- GPT-2 checkpoint loading
- Autoregressive text generation
- Temperature and top-k sampling

---

## License

This repository is intended for educational and research purposes.

GPT-2 weights and tokenizer assets remain subject to their respective OpenAI licenses.