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
