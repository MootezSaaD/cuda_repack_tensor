from torch.utils.cpp_extension import load
from naive_impl import repack_tensor_and_create_mask 
import numpy as np

import os, time, logging, torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
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

def warm_up_gpu(input):
    logger.info(f"Warm up...")
    input_tensor, mask = input
    for _ in range(10):
        output = fused_repack.fused_repack_merge(input_tensor, mask)
        torch.cuda.synchronize()
        _output = repack_tensor_and_create_mask(input_tensor, mask)

def benchmark_cuda(input, repeats=100):
    logger.info(f"Benchmarking CUDA implementation...")
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    input_tensor, mask = input
    
    times_ms = []
    for _ in range(repeats):
        start_event.record()
        output = fused_repack.fused_repack_merge(input_tensor, mask)
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    
    times_tensor = torch.tensor(times_ms)
    return times_tensor.mean().item(), times_tensor.std().item()


def benchmark_pytorch(input, repeats=100):
    logger.info(f"Benchmarking pytorch implementation...")
    input_tensor, mask = input

    times_ms = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = repack_tensor_and_create_mask(input_tensor, mask)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)  # Convert to ms
    
    return np.mean(times_ms), np.std(times_ms)

if __name__ == "__main__":

    input_tensor = torch.randn(64, 512, 768, device='cuda')
    mask = torch.randint(0, 2, (64, 512), device='cuda').bool()
    input = (input_tensor, mask)

    warm_up_gpu(input)
    
    torch.cuda.empty_cache()
    cuda_mean, cuda_std= benchmark_cuda(input)

    torch.cuda.empty_cache()
    pytorch_mean, pytorch_std = benchmark_pytorch(input)

    logging.info("\n=== Results ===")
    logging.info(f"CUDA:    {cuda_mean:.3f} ± {cuda_std:.3f} ms")
    logging.info(f"PyTorch: {pytorch_mean:.3f} ± {pytorch_std:.3f} ms")
    logging.info(f"Speedup: {pytorch_mean/cuda_mean:.1f}x")
