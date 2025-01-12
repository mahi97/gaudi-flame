import torch
import numpy as np


def network_to_flat(param_dict, return_meta=False):
    """
    Flattens a PyTorch state_dict (or any dict of tensors) into a 1D tensor.

    Returns:
      flat_vector: torch.Tensor of all parameters concatenated.
      meta: list of (key, shape, num_elements) to help re-build the dict later.
    """
    # Keep track of each parameter’s key, shape, and size
    meta = []
    chunks = []
    for key, tensor in param_dict.items():
        # Flatten the parameter
        flat_param = tensor.view(-1)
        chunks.append(flat_param)
        meta.append((key, tensor.shape, flat_param.numel()))

    # Concatenate everything into one big vector
    flat_vector = torch.cat(chunks)
    if return_meta:
        return flat_vector, meta
    return flat_vector


def flat_to_network(flat_vector, meta):
    """
    Rebuilds a parameter dictionary from a flat 1D tensor, given meta info.

    Returns:
      new_param_dict: a dictionary mapping keys to un-flattened tensors.
    """
    new_param_dict = {}
    offset = 0
    for (key, shape, size) in meta:
        # Slice out the appropriate chunk
        chunk = flat_vector[offset:offset + size]
        offset += size
        # Reshape to the original shape
        new_param_dict[key] = chunk.view(shape)
    return new_param_dict


# ----------------------------------------------------------
# Usage demonstration with an arbitrary PyTorch model
if __name__ == "__main__":
    import torch.nn as nn

    # Build a random model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )

    # Extract state_dict
    state_dict = model.state_dict()

    # Flatten
    flat_vector, meta_info = network_to_flat(state_dict)
    print("Flattened shape:", flat_vector.shape)

    # Restore
    new_dict = flat_to_network(flat_vector, meta_info)

    # Confirm the shapes match
    for k in state_dict.keys():
        orig_shape = state_dict[k].shape
        restored_shape = new_dict[k].shape
        print(f"{k}: original {orig_shape}, restored {restored_shape}")

    # (Optional) Load back into the model
    model.load_state_dict(new_dict)
