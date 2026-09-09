"""Portable optional NVIDIA telemetry; unavailable fields remain null."""
import csv
import io
import shutil
import subprocess
import threading
import time


class GPUMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_sample = 0
        self.cached = {}

    def sample(self):
        with self.lock:
            if time.monotonic() - self.last_sample < 2:
                return dict(self.cached)
            data = {'gpu_name': None, 'gpu_utilization_percent': None, 'gpu_power_watts': None,
                    'gpu_power_limit_watts': None, 'gpu_temperature_c': None,
                    'gpu_memory_used_mb': None, 'gpu_memory_total_mb': None, 'gpu_source': 'unavailable'}
            executable = shutil.which('nvidia-smi')
            if executable:
                try:
                    result = subprocess.run([executable, '--query-gpu=name,utilization.gpu,power.draw,power.limit,temperature.gpu,memory.used,memory.total',
                                             '--format=csv,noheader,nounits', '--id=0'], capture_output=True, text=True, timeout=2,
                                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    if result.returncode == 0:
                        row = next(csv.reader(io.StringIO(result.stdout)))
                        data['gpu_name'] = row[0].strip()
                        fields = ['gpu_utilization_percent', 'gpu_power_watts', 'gpu_power_limit_watts',
                                  'gpu_temperature_c', 'gpu_memory_used_mb', 'gpu_memory_total_mb']
                        for key, value in zip(fields, row[1:]):
                            try:
                                data[key] = float(value.strip())
                            except ValueError:
                                pass
                        data['gpu_source'] = 'nvidia-smi'
                except (OSError, subprocess.TimeoutExpired, StopIteration):
                    pass
            self.last_sample, self.cached = time.monotonic(), data
            return dict(data)
