

---

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