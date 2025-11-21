import torch as T

class GPUMemoryManager:
    @staticmethod
    def clear_cache():
        """Clear GPU cache periodically"""
        if T.cuda.is_available():
            T.cuda.empty_cache()
    
    @staticmethod
    def get_memory_info():
        """Monitor GPU memory usage"""
        if T.cuda.is_available():
            allocated = T.cuda.memory_allocated() / 1024**3
            cached = T.cuda.memory_reserved() / 1024**3
            return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
        return "GPU not available"
    

def setup_gpu_optimizations():
    if T.cuda.is_available():
        # Enable TF32 for faster matrix multiplications (Ampere+ GPUs)
        T.backends.cuda.matmul.allow_tf32 = True
        T.backends.cudnn.allow_tf32 = True
        
        # Other optimizations
        T.backends.cudnn.benchmark = True
        T.backends.cudnn.deterministic = False  # Set to True if reproducibility is needed
        
        # Increase GPU memory allocation growth
        T.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory


def adaptive_batch_sizing():
    """Dynamically adjust batch size based on available memory"""
    if T.cuda.is_available():
        total_memory = T.cuda.get_device_properties(0).total_memory
        available_memory = total_memory - T.cuda.memory_allocated()
        
        # Estimate memory per sample (adjust based on your model)
        memory_per_sample = 1024 * 1024  # 1MB per sample (adjust empirically)
        
        max_batch_size = int(available_memory * 0.7 / memory_per_sample)  # Use 70% of available memory
        return max(1, min(max_batch_size, 1024))  # Limit maximum batch size
    else:
        return 32  # Default batch size if GPU is not available