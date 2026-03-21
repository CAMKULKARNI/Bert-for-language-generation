import streamlit as st
from flax import nnx
from transformers import AutoTokenizer
import jax.numpy as jnp

from classes import BERTForCausalLM
from utils import load_model_weights

# Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class Config:
    vocab_size = tokenizer.vocab_size
    max_length = 64
    pad_token_id = tokenizer.pad_token_id
    batch_size = 16
    num_layers = 6
    d_model = 768
    num_heads = 12

config = Config()

# --- UI Setup ---
st.set_page_config(page_title="BERT based language Generator", layout="centered")
st.title("BERT Causal LM")
st.markdown("Generating text autoregressively using a bidirectional encoder.")

# --- Sidebar ---
st.sidebar.header("Inference Parameters")
max_new_tokens = st.sidebar.slider("Max Tokens", 10, 64, 30)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8, step=0.1)

@st.cache_resource
def load_model():
    """Loads and caches the BERT model."""
    model = BERTForCausalLM(
        vocab_size=config.vocab_size, 
        max_seq_len=config.max_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        rngs=nnx.Rngs(0)
    )

    # Dummy pass to initialize XLA graph
    dummy_input = jnp.zeros((1, config.max_length), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, config.max_length), dtype=jnp.int32)
    _ = model(dummy_input, dummy_mask)

    load_model_weights(model, "weights")
    return model

model = load_model()

# --- Main Interface ---
test_sentence = st.text_area("Enter your prompt:")

if st.button("Generate Text", type="primary"):
    encoded_inputs = tokenizer(
        test_sentence, 
        max_length=config.max_length,
        truncation=True,
        return_tensors="np",
        add_special_tokens=False
    )

    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']

    placeholder = st.empty()
    
    # Track raw IDs, not decoded strings, to handle WordPiece '##' correctly
    generated_token_ids = [] 

    # Stream the output
    for token in model.generate(input_ids, attention_mask, max_new_tokens, temperature=temperature, greedy=False, yield_token=True):
        # Extract the scalar integer from the (batch, 1) tensor
        token_id = int(token[0][0])
        generated_token_ids.append(token_id)
        
        # Decode the entire sequence at once
        decoded_generation = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        # Render markdown dynamically
        placeholder.markdown(f"**Output:**\n\n> {test_sentence} {decoded_generation}")