import os
import gc

# # Optional but highly recommended for 4GB GPUs:
# # If you DO want to pre-allocate to keep the speed benefits, but want to cap it to a safe limit (e.g., 50%):
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.90'

# # The equivalent of tf.config.experimental.set_memory_growth(gpu, True)
# # This stops XLA from grabbing 90% of your VRAM upfront. It will allocate memory only as needed.
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# # The equivalent of TF_GPU_ALLOCATOR='cuda_malloc_async'
# # This forces XLA to use the standard CUDA memory allocator instead of its internal BFC allocator.
# # It can be slower, but it plays much nicer if you have other processes using the GPU.
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


from flax import nnx
import optax
import orbax.checkpoint as ocp
from tqdm import tqdm # For a nice progress bar
from transformers import AutoTokenizer
from datasets import load_dataset

from classes import BERTForCausalLM
from utils import dynamic_batch_generator, save_model_weights, load_model_weights

print("Downloading and loading TinyStories...")
train_dataset = load_dataset("roneneldan/TinyStories", split="train")
val_dataset = load_dataset("roneneldan/TinyStories", split="validation")

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

test_sentence = "Dormammu, I've come to bargain!!"
encoded_inputs = tokenizer(
    test_sentence, 
    padding="max_length", 
    truncation=True, 
    max_length=config.max_length, # Kept short for this example
    return_tensors="np" 
)

next_token_id = model(encoded_inputs['input_ids'], encoded_inputs['attention_mask'])
# Print the tabular summary
print(nnx.tabulate(model, encoded_inputs['input_ids'], mask=encoded_inputs['attention_mask'], compute_flops=True))

train_steps = 5000
val_steps = 500


decay_end_step = 40_00_000
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=5e-4,
    warmup_steps=50_000,
    decay_steps=decay_end_step,
    end_value=1e-5
)


optimizer_chain = optax.chain(
    optax.clip_by_global_norm(1.0), # The safety net: Clip gradients if their norm exceeds 1.0
    optax.adamw(learning_rate=lr_schedule)
)


optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)


# Calculates the Cross Entropy Loss between our next-token predictions and the actual labels
def loss_fn(model, input_ids, attention_mask, target_ids):
    # Forward pass through our causal BERT head
    logits = model(input_ids, mask=attention_mask)
    
    # optax expects logits and integer labels. We take the mean loss over the batch.
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, 
        labels=target_ids
    ).mean()
    
    return loss


# This decorator compiles the entire forward pass, backward pass, and weight update into one ultra-fast GPU kernel
@nnx.jit
def train_step(model, optimizer, input_ids, attention_mask, target_ids):
    # nnx.value_and_grad computes the loss AND the gradients simultaneously
    loss, grads = nnx.value_and_grad(loss_fn)(model, input_ids, attention_mask, target_ids)
    
    # Update the model parameters
    optimizer.update(model, grads)
    
    return loss


@nnx.jit
def val_step(model, input_ids, attention_mask, target_ids):
    # nnx.value_and_grad computes the loss AND the gradients simultaneously
    loss = loss_fn(model, input_ids, attention_mask, target_ids)

    return loss


# 1. Define the checkpoint directory
checkpoint_dir = os.path.abspath("bert_causal_lm_checkpoints")

# 2. Set up the Checkpoint Manager
# We tell it to only keep the 2 most recent checkpoints to save disk space
options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)

# 3. Restoration Logic
start_step = 0
if checkpoint_manager.latest_step() is not None:
    print(f"Found existing checkpoint at step {checkpoint_manager.latest_step()}! Restoring...")
    
    # Extract the abstract state to tell Orbax what shapes to expect
    abstract_state = nnx.state((model, optimizer))
    
    # Restore the saved state
    restored_state = checkpoint_manager.restore(
        checkpoint_manager.latest_step(), 
        args=ocp.args.StandardRestore(abstract_state)
    )
    
    # Update BOTH the model and the optimizer with the restored weights and momentum
    nnx.update((model, optimizer), restored_state)
    
    # Destroy the temporary checkpoint object and force garbage collection!
    del restored_state
    gc.collect()

    # Set the starting step so our progress bar is accurate
    start_step = checkpoint_manager.latest_step()
else:
    print("No checkpoint found. Starting training from scratch.")

if start_step == 0:
    file_to_delete = "train.log"
    # Check if the file exists before attempting to delete
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)
        print(f"File '{file_to_delete}' deleted successfully.")
    else:
        print(f"File '{file_to_delete}' does not exist.")
    file_to_delete = "val.log"
    # Check if the file exists before attempting to delete
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)
        print(f"File '{file_to_delete}' deleted successfully.")
    else:
        print(f"File '{file_to_delete}' does not exist.")

# Initialize a global step counter based on where we restored from
global_step = start_step

train_dataset = train_dataset.shuffle(seed=start_step % 1319)
val_dataset = val_dataset.shuffle(seed=start_step % 1319)

# Initialize our lazy generator
train_dataset_stream = dynamic_batch_generator(train_dataset, tokenizer, config.batch_size, config.max_length)
val_dataset_stream = dynamic_batch_generator(val_dataset, tokenizer, config.batch_size, config.max_length)

while True:
    train_losses = [] # Store losses in fast RAM
    
    with tqdm(total=train_steps, desc="Training Steps") as pbar:
        for step in range(train_steps):
            batch_inputs, batch_masks, batch_labels = next(train_dataset_stream)
            
            loss = train_step(model, optimizer, batch_inputs, batch_masks, batch_labels)
            
            # Append to list instead of writing to disk
            train_losses.append(f"{float(loss):.4f}\n")

            global_step += 1
            pbar.set_postfix({'loss': f"{float(loss):.4f}"})
            pbar.update(1)
            
    # --- BATCH I/O WRITE ---
    # Open the file exactly ONCE per epoch and dump all 5000 lines at once
    with open("train.log", "a") as file:
        file.writelines(train_losses)
            
    # --- SAVING LOGIC ---
    current_state = nnx.state((model, optimizer))
    checkpoint_manager.save(global_step, args=ocp.args.StandardSave(current_state))
    checkpoint_manager.wait_until_finished()

    save_model_weights(model, "weights")
    print(f"\nModel and Optimizer saved at global step {global_step}!")

    val_losses = []
    with tqdm(total=val_steps, desc="Val Steps") as pbar:
        for step in range(val_steps):
            batch_inputs, batch_masks, batch_labels = next(val_dataset_stream)
            loss = val_step(model, batch_inputs, batch_masks, batch_labels)
            
            val_losses.append(f"{float(loss):.4f}\n")
            pbar.set_postfix({'loss': f"{float(loss):.4f}"})
            pbar.update(1)
            
    # Dump validation logs
    with open("val.log", "a") as file:
        file.writelines(val_losses)
    
    with open("train.log", "a") as file:
        file.write("\n")
    with open("val.log", "a") as file:
        file.write("\n")
        
    # --- FORCE RAM CLEANUP ---
    # Manually destroy the dead objects and force the Garbage Collector to run
    del train_losses, val_losses, batch_inputs, batch_masks, batch_labels
    gc.collect()