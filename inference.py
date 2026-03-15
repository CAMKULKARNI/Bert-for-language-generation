import os

# THE MAGIC LINE: Force JAX to run exclusively on the CPU!
# THIS MUST BE THE VERY FIRST THING IN THE SCRIPT.
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
from flax import nnx
import jax.numpy as jnp
from tqdm import tqdm # For a nice progress bar
from transformers import AutoTokenizer

from classes import BERTForCausalLM
from utils import dynamic_batch_generator, save_model_weights, load_model_weights

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
model = BERTForCausalLM(
    vocab_size=config.vocab_size, 
    max_seq_len=config.max_length,
    num_layers=config.num_layers,
    d_model=config.d_model,
    num_heads=config.num_heads,
    rngs=nnx.Rngs(0)
)

# test_sentence = "Once upon a time, there was a little girl named Lily. She loved to play with her"
# test_sentence = "Timmy was very happy. He found a big red"
test_sentence = "The little dog ran into the park and saw a"
encoded_inputs = tokenizer(
    test_sentence, 
    max_length=config.max_length,
    truncation=True,
    return_tensors="np",
    add_special_tokens=False
)

next_token_id = model(encoded_inputs['input_ids'], encoded_inputs['attention_mask'])
# # Print the tabular summary
# print(nnx.tabulate(model, encoded_inputs['input_ids'], mask=encoded_inputs['attention_mask'], compute_flops=True))

load_model_weights(model, "weights")

max_new_tokens = 30
input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']

key = jax.random.PRNGKey(1319) 
temperature = 0.8 # The industry-standard sweet spot for generation

for _ in tqdm(range(max_new_tokens)):
    # Slice to strictly enforce the sliding window limit (max 64 tokens)
    curr_input_ids = input_ids[:, -config.max_length:]
    curr_attention_mask = attention_mask[:, -config.max_length:]
    
    # Get predictions
    logits = model(curr_input_ids, mask=curr_attention_mask)
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Split the key (JAX requires a fresh random key for every single random operation)
    key, subkey = jax.random.split(key)
    
    # Sample from the probability distribution instead of using argmax
    next_token = jax.random.categorical(subkey, scaled_logits, axis=-1)
    
    # Categorical returns a 1D array (batch,), so we expand it to (batch, 1) for concatenation
    next_token = jnp.expand_dims(next_token, axis=-1)
    
    # Safely concatenate
    input_ids = jnp.concatenate([input_ids, next_token], axis=1)
    attention_mask = jnp.concatenate([attention_mask, jnp.ones((1, 1), dtype=jnp.int32)], axis=1)

print("\n--- Generated Output ---")
print(tokenizer.decode(input_ids[0], skip_special_tokens=True))