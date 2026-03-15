from flax import nnx
import jax 
import jax.numpy as jnp


class EncoderLayer(nnx.Module):
    def __init__(self, d_model, num_heads, *, rngs: nnx.Rngs):
        super(EncoderLayer, self).__init__()

        self.mha = nnx.MultiHeadAttention(num_heads, d_model, rngs=rngs, use_bias=False, decode=False)
        self.linear1 = nnx.Linear(d_model, d_model * 4, use_bias=False, rngs=rngs)
        self.linear2 = nnx.Linear(d_model * 4, d_model, use_bias=False, rngs=rngs)
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.swish = nnx.swish

    def __call__(self, x, mask=None):
        x_norm1 = self.norm1(x)
        attn_output = self.mha(x_norm1, x_norm1, x_norm1, mask=mask)
        out1 = x + attn_output

        out1_norm = self.norm2(out1)
        linear_output = self.linear2(self.swish(self.linear1(out1_norm)))
        out2 = out1 + linear_output

        return out2


class BERTBackBone(nnx.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers, d_model, num_heads, *, rngs: nnx.Rngs):
        super(BERTBackBone, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.token_embeddings = nnx.Embed(self.vocab_size, d_model, rngs=rngs)
        self.pos_embeddings = nnx.Embed(self.max_seq_len, d_model, rngs=rngs)
        self.layers = nnx.List([EncoderLayer(self.d_model, self.num_heads, rngs=rngs) for _ in range(self.num_layers)])

    def __call__(self, ids, mask=None):
        tok_emb = self.token_embeddings(ids)
        pos_emb = self.pos_embeddings(jnp.arange(ids.shape[1]))
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x, mask=mask)

        return x


class BERTForCausalLM(nnx.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers, d_model, num_heads, *, rngs: nnx.Rngs):
        super(BERTForCausalLM, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.bert_backbone = BERTBackBone(self.vocab_size, self.max_seq_len, self.num_layers, self.d_model, self.num_heads, rngs=rngs)

        # Projection layer (768 -> 30522)
        self.out_proj = nnx.Linear(self.d_model, vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, ids, mask=None):
        # 1. Figure out the index of the actual last token BEFORE we alter the mask shape
        if mask is not None:
            # Sum the 1s to get the length, subtract 1 to get the final valid index
            last_token_indices = jnp.sum(mask, axis=-1) - 1
            
            # Transforms (batch, seq_len) -> (batch, 1, 1, seq_len) for the attention mechanism
            extended_mask = mask[:, None, None, :]
        else:
            extended_mask = None
            # If there is no mask, we assume the window is full, so the last token is at the very end
            last_token_indices = jnp.full((ids.shape[0],), ids.shape[1] - 1)

        # 2. Run the forward pass
        x = self.bert_backbone(ids, mask=extended_mask)
        
        # 3. Use JAX advanced indexing to dynamically pluck the correct token out of each row!
        batch_indices = jnp.arange(x.shape[0])
        last_token = x[batch_indices, last_token_indices, :]
        
        # 4. Project that specific token to the vocabulary
        logits = self.out_proj(last_token)

        return logits

    def generate(self, ids, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.max_seq_len:])
            next_token = jnp.argmax(logits, axis=-1, keepdims=True)
            ids = jnp.concatenate([ids, next_token], axis=1)

        return ids