import jax.numpy as jnp
import numpy as np
import os
import orbax.checkpoint as ocp
from flax import nnx


def dynamic_batch_generator(dataset, tokenizer, batch_size, max_length):
    batch_inputs, batch_masks, batch_labels = [], [], []
    
    # Iterate continuously through the streamed dataset
    for example in dataset:
        # Tokenize the single story on the fly
        tokenized = tokenizer(example["text"], add_special_tokens=False)
        ids = tokenized["input_ids"]
        
        if len(ids) > 1:
            # Extract sliding windows from this story
            for i in range(len(ids) - 1):
                chunk = ids[i : i + max_length + 1]
                max_pos_ctx = len(chunk) - 1
                
                # Dynamic Uniform Masking
                ctx_len = np.random.randint(1, max_pos_ctx + 1)
                
                input_window = np.zeros(max_length, dtype=np.int32)
                input_window[:ctx_len] = chunk[:ctx_len]
                
                mask_window = np.zeros(max_length, dtype=np.int32)
                mask_window[:ctx_len] = 1
                
                target = chunk[ctx_len]
                
                batch_inputs.append(input_window)
                batch_masks.append(mask_window)
                batch_labels.append(target)
                
                # When our batch is full, yield it as JAX arrays and clear the lists!
                if len(batch_inputs) == batch_size:
                    yield (
                        jnp.array(batch_inputs),
                        jnp.array(batch_masks),
                        jnp.array(batch_labels)
                    )
                    batch_inputs, batch_masks, batch_labels = [], [], []


def save_model_weights(model, folder_path):
    """Extracts purely the model weights and saves them to a directory."""
    model_state = nnx.state(model)
    checkpointer = ocp.StandardCheckpointer()
    
    checkpointer.save(
        os.path.abspath(folder_path), 
        model_state,
        force=True
    )
    
    print(f"Model weights successfully saved to: {folder_path}/")


def load_model_weights(model, folder_path):
    """Loads weights from a directory directly into the provided model."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Weight folder '{folder_path}' does not exist.")
        
    checkpointer = ocp.StandardCheckpointer()
    
    # 1. Extract the 'abstract state' from the current model. 
    # This acts as a blank template so Orbax knows exactly what tensor shapes to load.
    abstract_state = nnx.state(model)
    
    # 2. Restore the state from the folder using the template
    restored_state = checkpointer.restore(
        os.path.abspath(folder_path), 
        abstract_state,
        strict=True
    )
    
    # 3. Inject the loaded weights directly back into the model
    nnx.update(model, restored_state)
    
    print(f"Model weights successfully loaded from: {folder_path}/")