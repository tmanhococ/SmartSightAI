import time
from src.utils.monitor import ExecutionMonitor

def test_execution_monitor():
    monitor = ExecutionMonitor()
    assert monitor.get_ram_usage() > 0.0
    
    with monitor.track("test_stage"):
        time.sleep(0.1)
        
    durations = monitor.get_durations()
    assert "test_stage" in durations
    assert durations["test_stage"] >= 0.1
