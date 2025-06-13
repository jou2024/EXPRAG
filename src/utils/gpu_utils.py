import pynvml
IS_DEBUG_GPU = False
def print_gpu_utilization(info=""):
    def _convert_bytes(num_bytes):
        for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if num_bytes < 1024:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024
    if IS_DEBUG_GPU:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = _convert_bytes(meminfo.total)
        free_memory = _convert_bytes(meminfo.free)
        used_memory = _convert_bytes(meminfo.used)

        print("---------------------------------------")
        print(info)
        print(f"    Total memory: {total_memory}")
        print(f"    Free memory: {free_memory}")
        print(f"    Used memory: {used_memory}")
        print("---------------------------------------")
        pynvml.nvmlShutdown()