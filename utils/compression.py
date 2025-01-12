import torch


def topk_sparsify(grad, ratio=0.01):
    """
    Keep only the top-`ratio` portion of largest elements by absolute value.
    The rest are set to zero.

    Args:
      grad (torch.Tensor): 1D tensor of gradients.
      ratio (float): fraction of non-zero entries to keep.

    Returns:
      grad_sparsified (torch.Tensor): same shape as grad, with most elements zeroed out.
      mask (torch.Tensor): boolean mask where True = retained elements.
    """
    if ratio <= 0 or ratio >= 1:
        raise ValueError("ratio must be in (0,1).")

    # Number of elements to keep
    k = int(ratio * grad.numel())
    # Get threshold by partial sort
    threshold = torch.kthvalue(grad.abs(), grad.numel() - k).values
    # Build mask
    mask = (grad.abs() >= threshold)
    # Zero out everything else
    grad_sparsified = grad * mask

    return grad_sparsified


def quantize(grad, bits=8):
    """
    Uniformly quantize the gradient into `bits`-bits fixed-point.

    Args:
      grad (torch.Tensor): 1D tensor of (possibly already sparsified) gradients.
      bits (int): number of bits for quantization.

    Returns:
      grad_quant (torch.Tensor): quantized gradient, still float, but with discrete levels.
      scale (float): scaling factor used for quantization.
      grad_min (float), grad_max (float): min and max values for potential dequant.
    """
    if bits < 1 or bits > 32:
        raise ValueError("bits must be in [1, 32] for this simple uniform quantization.")

    grad_min = grad.min()
    grad_max = grad.max()

    # Edge case: if everything is zero, skip
    if grad_max.item() == grad_min.item():
        return grad.clone(), 1.0, grad_min.item(), grad_max.item()

    # Number of discrete levels
    levels = 2 ** bits - 1

    # Scale to [0, levels]
    scale = (levels) / (grad_max - grad_min)
    grad_shifted = (grad - grad_min) * scale

    # Round to nearest integer
    grad_int = torch.round(grad_shifted).clamp(0, levels)

    # Scale back to original range as float
    grad_quant = grad_int / scale + grad_min

    return grad_quant


# -------------------------------------------------------------------------
# Usage example
if __name__ == "__main__":
    # Make a random gradient tensor
    g = torch.randn(1000)

    # 1) Sparsify
    g_sparse = topk_sparsify(g, ratio=0.05)  # keep top 5%

    # 2) Quantize
    g_quant = quantize(g_sparse, bits=8)

    print("Original non-zero:", (g != 0).sum().item())
    print("After top-k sparsification non-zero:", (g_sparse != 0).sum().item())
    # print("Quantization scale:", scale)
    # print("Min/Max used for quantization:", gmin, gmax)
