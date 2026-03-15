from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

max_length = 64

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

import numpy as np
from datasets import load_dataset

print("Downloading and loading TinyStories...")
train_dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
test_dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=False)

def create_dynamic_windows(examples):
    tokenized = tokenizer(examples["text"], add_special_tokens=False)
    
    input_windows = []
    attention_masks = []
    target_tokens = []
    
    for ids in tokenized["input_ids"]:
        # We need at least 2 tokens to have 1 input and 1 target
        if len(ids) > 1:
            # We slide across the story grabbing chunks up to 17 tokens long
            # (16 for max context + 1 for the target)
            for i in range(len(ids) - 1):
                # Grab a chunk of available tokens (up to 17)
                chunk = ids[i : i + max_length + 1]
                
                # Determine maximum possible context length for this specific chunk
                max_possible_context = len(chunk) - 1
                
                # YOUR IDEA: Randomly pick how much context to keep!
                context_len = np.random.randint(1, max_possible_context + 1)
                
                # 1. Create the padded input_ids (length 16, filled with 0s)
                input_ids = np.zeros(max_length, dtype=np.int32)
                input_ids[:context_len] = chunk[:context_len]
                
                # 2. Create the attention_mask (length 16, 1s for real tokens, 0s for padding)
                mask = np.zeros(max_length, dtype=np.int32)
                mask[:context_len] = 1
                
                # 3. The target is the very next token after our chosen context
                target = chunk[context_len]
                
                input_windows.append(input_ids)
                attention_masks.append(mask)
                target_tokens.append(target)
                
    return {
        "input_ids": input_windows, 
        "attention_mask": attention_masks, 
        "labels": target_tokens
    }

print("Chopping and dynamically masking stories...")
windowed_train_dataset = train_dataset.map(
    create_dynamic_windows, 
    batched=True, 
    remove_columns=["text"]
)

windowed_train_dataset.set_format(type="numpy")
print(f"Total dynamically masked windows created: {len(windowed_train_dataset)}")

# print(windowed_train_dataset.features["input_ids"])
# print(windowed_train_dataset.features["attention_mask"])
# print(windowed_train_dataset.features["labels"])

# # 5. Accessing individual elements
# print("\n--- Accessing Data ---")
# first_row = windowed_train_dataset[0]
# print(f"First window input_ids: {first_row['input_ids']}")
# print(f"First window attention_mask: {first_row['attention_mask']}")
# print(f"First window label: {first_row['labels']}")

# batch_slice = windowed_train_dataset[0:2]
# print(f"Slice input_ids shape: {batch_slice['input_ids'].shape}")

# counter = 0
# for x, mask, y in zip(batch_slice["input_ids"], batch_slice["attention_mask"], batch_slice["labels"]):
#     print("X: ", x)
#     print("mask: ", mask)
#     print("Y: ", y)
#     print()
#     counter += 1
#     if counter == 2: break

# windowed_test_dataset = test_dataset.map(
#     create_dynamic_windows, 
#     batched=True, 
#     remove_columns=["text"]
# )

# windowed_test_dataset.set_format(type="numpy")
# print(f"Total dynamically masked windows created: {len(windowed_test_dataset)}")

# print(windowed_test_dataset.features["input_ids"])
# print(windowed_test_dataset.features["attention_mask"])
# print(windowed_test_dataset.features["labels"])

# # 5. Accessing individual elements
# print("\n--- Accessing Data ---")
# first_row = windowed_test_dataset[0]
# print(f"First window input_ids: {first_row['input_ids']}")
# print(f"First window attention_mask: {first_row['attention_mask']}")
# print(f"First window label: {first_row['labels']}")

# batch_slice = windowed_test_dataset[0:2]
# print(f"Slice input_ids shape: {batch_slice['input_ids'].shape}")

# counter = 0
# for x, mask, y in zip(batch_slice["input_ids"], batch_slice["attention_mask"], batch_slice["labels"]):
#     print("X: ", x)
#     print("mask: ", mask)
#     print("Y: ", y)
#     print()
#     counter += 1
#     if counter == 2: break