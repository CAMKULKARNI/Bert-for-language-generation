import jax
from flax import nnx
import jax.numpy as jnp

def main():
    # 1. Check JAX backend and available devices
    print("--- JAX System Check ---")
    print(f"Default Backend: {jax.default_backend().upper()}")
    
    devices = jax.devices()
    print(f"Total Devices Found: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"Device {i}: {device.device_kind} (ID: {device.id})")

    # 2. Check Flax NNX Allocation
    print("\n--- Flax NNX Allocation Check ---")
    # Initialize a random number generator state
    rngs = nnx.Rngs(0)
    
    # Create a simple linear layer
    layer = nnx.Linear(in_features=32, out_features=64, rngs=rngs)
    
    # Check where the weights (kernel) are stored
    weight_device = layer.kernel.get_value().device
    print(f"NNX Layer weights allocated on: {weight_device}")
    
    # 3. Perform a quick computation
    x = jnp.ones((1, 32)) # Dummy input
    y = layer(x)
    print(f"Computation successful! Output shape: {y.shape}")

if __name__ == "__main__":
    main()