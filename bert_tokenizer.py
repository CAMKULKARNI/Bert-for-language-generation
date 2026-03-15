from transformers import AutoTokenizer

# 1. Load the tokenizer for BERT Base (uncased means it converts everything to lowercase)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. Extract the properties for our NNX model initialization
vocab_size = tokenizer.vocab_size
max_length = tokenizer.model_max_length
pad_token_id = tokenizer.pad_token_id

print(f"--- BERT Base Properties ---")
print(f"Vocabulary Size: {vocab_size}") # Usually 30,522
print(f"Max Sequence Length: {max_length}") # Usually 512
print(f"Pad Token ID: {pad_token_id}") # Usually 0

# 3. Tokenize a test sentence 
# We use return_tensors="np" (NumPy) because JAX natively consumes NumPy arrays!
test_sentence = "The mad scientist laughed."
encoded_inputs = tokenizer(
    test_sentence, 
    padding="max_length", 
    truncation=True, 
    max_length=10, # Kept short for this example
    return_tensors="np" 
)

print("\n--- Tokenizer Outputs ---")
print(f"Input IDs:\n{encoded_inputs['input_ids']}")
print(f"Token Type IDs (Segment):\n{encoded_inputs['token_type_ids']}")
print(f"Attention Mask:\n{encoded_inputs['attention_mask']}")