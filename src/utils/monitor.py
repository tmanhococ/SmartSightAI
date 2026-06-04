import psutil
import time
from contextlib import contextmanager

class ExecutionMonitor:
    def __init__(self):
        self.durations = {}
        
    def get_ram_usage(self) -> float:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # MB
        
    @contextmanager
    def track(self, stage_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[stage_name] = time.perf_counter() - start
            
    def get_durations(self) -> dict:
        return self.durations
