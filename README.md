# cuda_repack_tensor
A CUDA kernel that eliminates elements from a 3D tensor given a 2D mask.  
Tested on an RTX3070Ti with cuda 12.3, torch 2.0.1 and numpy 1.26.4.  

```
2025-01-23 16:47:38,552 - INFO - Warm up...
2025-01-23 16:47:38,716 - INFO - Benchmarking CUDA implementation...
2025-01-23 16:47:39,023 - INFO - Benchmarking pytorch implementation...
2025-01-23 16:47:40,188 - INFO - 
=== Results ===
2025-01-23 16:47:40,189 - INFO - CUDA:    2.838 ± 0.144 ms
2025-01-23 16:47:40,189 - INFO - PyTorch: 11.641 ± 0.507 ms
2025-01-23 16:47:40,189 - INFO - Speedup: 4.1x

```
