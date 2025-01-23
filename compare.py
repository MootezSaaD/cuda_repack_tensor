import torch
import logging
import os
from typing import Tuple


from torch.utils.cpp_extension import load
from naive_impl import repack_tensor_and_create_mask as pytorch_implementation



os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6' # You may need to change this. This was used for an RTX 3070Ti.
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load your CUDA extension
fused_repack = load(
    name='fused_repack',
    sources=['fused_repack.cpp', 'fused_repack.cu'],
    extra_cflags=['-O3', '-fno-strict-aliasing'],
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-std=c++17', 
        '-Xcompiler=-fno-strict-aliasing'
    ]
)

def compare_methods(
    input: torch.Tensor,
    mask: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    verbose: bool = False
) -> Tuple[bool, bool]:
    """
    Compare CUDA and PyTorch implementations for both data and mask outputs.
    
    Args:
        input: Input tensor (B, N, d)
        mask: Boolean mask tensor (B, N)
        atol: Absolute tolerance
        rtol: Relative tolerance
        verbose: Print full mismatch details
    
    Returns:
        (data_match, mask_match) tuple of booleans
    """
    # Generate outputs from both implementations
    cuda_out, cuda_mask = fused_repack.fused_repack_merge(input, mask)
    pytorch_out, pytorch_mask = pytorch_implementation(input, mask, True)  # Your PyTorch version
    
    # Ensure we're comparing on the same device
    pytorch_out = pytorch_out.to(cuda_out.device)
    pytorch_mask = pytorch_mask.to(cuda_mask.device)

    # Check tensor shapes first
    shape_match = cuda_out.shape == pytorch_out.shape and cuda_mask.shape == pytorch_mask.shape
    if not shape_match:
        logging.error(f"Shape mismatch - CUDA: {cuda_out.shape}, PyTorch: {pytorch_out.shape}")
        return False, False

    # Compare data tensors with tolerance
    data_close = torch.allclose(cuda_out, pytorch_out, atol=atol, rtol=rtol)
    
    # Compare boolean masks exactly
    mask_equal = torch.equal(cuda_mask, pytorch_mask)

    # Detailed error reporting
    if not data_close or not mask_equal:
        logging.error("Validation failed:")
        
        if not data_close:
            max_diff = torch.max(torch.abs(cuda_out - pytorch_out)).item()
            logging.error(f"Data max difference: {max_diff:.2e}")
            
            if verbose:
                diff_mask = torch.abs(cuda_out - pytorch_out) > atol
                logging.error(f"Mismatch locations:\n{diff_mask.nonzero()}")
                logging.error(f"CUDA values:\n{cuda_out[diff_mask]}")
                logging.error(f"PyTorch values:\n{pytorch_out[diff_mask]}")

        if not mask_equal:
            mismatch_count = torch.sum(cuda_mask != pytorch_mask).item()
            logging.error(f"Mask mismatches: {mismatch_count}/{cuda_mask.numel()}")
            
            if verbose:
                logging.error(f"Mismatch indices:\n{(cuda_mask != pytorch_mask).nonzero()}")

    return data_close, mask_equal

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create test input
    B, N, d = 2, 5, 3
    input = torch.randn(B, N, d, device="cuda")
    mask = torch.randint(0, 2, (B, N), device="cuda").bool()

    # Run comparison
    data_ok, mask_ok = compare_methods(input, mask, verbose=True)
    
    if data_ok and mask_ok:
        logging.info("All outputs match!")
    else:
        logging.error("Output mismatch detected")
        # Optional: Save problematic tensors for debugging
        torch.save({"input": input, "mask": mask}, "debug_case.pt")