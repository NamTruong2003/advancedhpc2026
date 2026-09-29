from numba import cuda

gpu = cuda.get_current_device()

free_memory, total_memory = cuda.current_context().get_memory_info()

print("=== GPU Information ===")
print("Device name:", gpu.name)
print("Multiprocessor count:", gpu.MULTIPROCESSOR_COUNT)
print("Memory size:", total_memory, "bytes")
print("Memory size:", total_memory / (1024**3), "GB")
print("Compute capability:", gpu.compute_capability)