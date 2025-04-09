# Transformer from Scratch - Character-Level Language Model (Shakespeare)

This project implements a Transformer-based character-level language model from scratch using PyTorch. It is trained on the Tiny Shakespeare dataset to generate Shakespearean-style text.

## Overview
This project is inspired by Andrej Karpathy's work on character-level RNNs and modern Transformer architectures. It features:

- Pure PyTorch implementation (no external libraries like HuggingFace)

- Multi-head self-attention with masked causal attention

- Positional encoding (both learnable and sinusoidal supported)

- LayerNorm, feed-forward blocks, dropout regularization

- Character-level tokenization

- Training loop and generation script for text synthesis

## Dataset

Name: Tiny Shakespeare
Size: ~1MB

Type: Plain text with character-level granularity

Download the dataset using:
```wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt```


## Model Architecture:

[Input] 
   → [Embedding + Positional Encoding] 
   → [Transformer Blocks × N] 
   → [LayerNorm] 
   → [Linear Output Head] 
   → [Softmax Probabilities]

- Token Embedding: Character-level

- Positional Encoding: Learnable (optionally sinusoidal)

- Transformer Blocks: Configurable (default: 4 layers)

- Self-Attention: Multi-head (default: 4 heads)

- Feedforward Network: 2-layer MLP with ReLU

- Final Layer: Linear layer projecting to vocabulary size

## ⚙️ Hyperparameters

| Parameter       | Value  |
|----------------|--------|
| `block_size`   | 128    |
| `batch_size`   | 64     |
| `embed_dim`    | 128    |
| `num_heads`    | 4      |
| `num_layers`   | 4      |
| `ff_hidden_dim`| 512    |
| `learning_rate`| 3e-4   |
| `max_iters`    | 10000  |



## Training

Run the training loop with:
```python char_transformer.py```
Optimizer: Adam

Loss Function: CrossEntropyLoss

## Text Generation
Generate new Shakespeare-like text using a prompt:

python

```
start = encode("The king")
out = generate(model, start, max_new_tokens=100)
print(decode(out))
```
Example Output:
```
The king, deaths, come the duke thee?

GLOUCESTER:
Madam, though I wash a wagmany, and sensel in those,
Make
```

## Future Improvements
- Add temperature & top-k/top-p sampling

- Add learning rate scheduler (e.g., cosine annealing)

- Save and load model checkpoints

- Evaluate perplexity on validation set

- Extend to word-level tokenization or Byte Pair Encoding (BPE)
