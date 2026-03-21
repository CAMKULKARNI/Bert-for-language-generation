import os

# THE MAGIC LINE: Force JAX to run exclusively on the CPU!
# THIS MUST BE THE VERY FIRST THING IN THE SCRIPT.
os.environ["JAX_PLATFORMS"] = "cpu"

from flax import nnx
from transformers import AutoTokenizer

from classes import BERTForCausalLM
from utils import load_model_weights

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

max_new_tokens = 50
input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']

temperature = 0.8
output = model.generate(input_ids, attention_mask, max_new_tokens, temperature=temperature, greedy=False)

print("\n--- Generated Output ---")
print(tokenizer.decode(output[0], skip_special_tokens=True))