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

---

### Quick Start

#### 1. Clone the repository:
```bash
git clone [https://github.com/ArisPagonopoulos/gpt2-from-scratch-pytorch.git](https://github.com/ArisPagonopoulos/gpt2-from-scratch-pytorch.git)
cd gpt2-from-scratch-pytorch```

#### 2. Install dependencies
```bash
pip install -r requirements.txt

#### 3. Generate text with pre-trained GPT-2 (124M) weights:
```bash
python gpt_custom_model.py "Every journey begins with" --loaded --tokens 30 --temperature 0.8 --top_k 40

Or run with randomly initialized weights:
'''bash
python gpt_custom_model.py "Every journey begins with" --tokens 20
