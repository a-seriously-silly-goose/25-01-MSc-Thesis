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