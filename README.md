# BERT for Causal Language Generation

This repository contains a custom implementation of a Causal Language Model (CLM) built on top of a BERT-like bidirectional architecture. It leverages the high-performance JAX and Flax (specifically the new `nnx` API) libraries for training and inference. The model is trained on the `TinyStories` dataset to generate coherent text autoregressively.

## Links & Resources

- **Model Weights**:  ([Download here](https://drive.google.com/drive/folders/1QNjlbvicjo7TOA4wOuKxL9qNpCvb9QCu?usp=sharing)). *Note: The weights will need to be downloaded from the link into a directory named `weights` in the same directory as all the other files.*
- **Medium Blog Post**: Read the full guide ([Medium Blog](https://medium.com/@camkoolkarni/forcing-a-bidirectional-encoder-to-decode-using-bert-to-generate-language-aa1fab6dd4d3))
- **YouTube Video**: Watch the tutorial (Coming Soon)

## Architectural Overview

Unlike typical autoregressive models (like GPT) that use decoder-only architectures with causal masking, this project adapts a **bidirectional encoder** (BERT) for next-token prediction.

- **Model Backbone**: The `BERTBackBone` consists of standard Transformer encoder layers (`MultiHeadAttention`, `LayerNorm`, Feed-Forward networks with `swish` activation).
- **Next-Token Prediction**: During training, dynamic windows of context are sampled from the text. The model reads the bidirectional context and projects the *last* token's representation to the vocabulary dimension to predict the subsequent target token.
- **Inference/Generation**: For generation, the model re-encodes the entire sequence (up to a maximum sequence length, operating as a sliding window) at each step. While more computationally expensive than KV-caching decoder models, it leverages full bidirectional context for the generated prompt before predicting the next word.
- **Dataset**: Trained on the `roneneldan/TinyStories` dataset, which is well-suited for training small, specialized language models.
- **Tokenizer**: Uses the standard `bert-base-uncased` WordPiece tokenizer from Hugging Face.

## Code Structure

- **`classes.py`**: Defines the core neural network architecture using Flax `nnx`. Includes `EncoderLayer`, `BERTBackBone`, and the `BERTForCausalLM` wrapper with a custom autoregressive `generate` method.
- **`train.py`**: The main training loop. Handles data loading, distributed training configuration, learning rate scheduling (warmup with cosine decay), parameter optimization via Optax, checkpointing via Orbax, and logs progress.
- **`utils.py`**: Contains helper functions, including the `dynamic_batch_generator` (which creates sliding windows of text and dynamic length masking on the fly) and save/load utilities for Orbax checkpoints.
- **`inference.py`**: A standalone script for text generation. Configured to run exclusively on the CPU for broad compatibility and easy testing.
- **`app.py`**: A Streamlit web application providing a user-friendly interface for interacting with the trained model and dynamically streaming text generation.
- **`dataset_check.py`**: A utility script to inspect and debug the dynamic window dataset creation and masking process.
- **`check_hardware.py`**: A diagnostic script to verify JAX backend configurations and ensure proper device allocation (e.g., GPU vs CPU).
- **`bert_tokenizer.py`**: A simple script to explore the properties, vocabulary sizes, and outputs of the `bert-base-uncased` tokenizer.

## Getting Started

### Prerequisites

Ensure you have the required libraries installed. You will need `jax`, `flax`, `optax`, `orbax-checkpoint`, `transformers`, `datasets`, and `streamlit`. 

Depending on your hardware, ensure you install the appropriate JAX version (CPU, CUDA, or TPU) to utilize hardware acceleration.

### Training the Model

To start training the model from scratch (or resume from a checkpoint if one exists), run:

```bash
python train.py
```

This script will:
1. Download the TinyStories dataset.
2. Initialize the Flax NNX model and compile the ultra-fast JAX training step kernels.
3. Begin training, logging loss progress to `train.log` and `val.log`.
4. Automatically save standard and abstract checkpoints using Orbax.

### Running Inference

You can test the generation capabilities via the terminal:

```bash
python inference.py
```
*(Note: `inference.py` is configured to force CPU execution by default via environment variables).*

### Streamlit Web App

To interact with the model via a clean UI, run the Streamlit app:

```bash
streamlit run app.py
```

This will launch a local web server where you can enter text prompts, adjust the temperature, modify the maximum number of new tokens, and watch the model stream its output in real-time.