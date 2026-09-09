from types import SimpleNamespace
from app.gpu_metrics import GPUMetrics


def test_measured_power_and_unsupported_limit(monkeypatch):
    monkeypatch.setattr('app.gpu_metrics.shutil.which', lambda _: 'nvidia-smi')
    calls = []
    def run(*args, **kwargs):
        calls.append(args)
        assert kwargs['timeout'] == 2
        return SimpleNamespace(returncode=0, stdout='NVIDIA Test GPU, 14, 9.02, [N/A], 52, 815, 8151\n')
    monkeypatch.setattr('app.gpu_metrics.subprocess.run', run)
    monitor = GPUMetrics()
    result = monitor.sample()
    assert result['gpu_power_watts'] == 9.02
    assert result['gpu_power_limit_watts'] is None
    assert result['gpu_utilization_percent'] == 14
    assert result['gpu_name'] == 'NVIDIA Test GPU'
    assert monitor.sample() == result and len(calls) == 1


def test_missing_gpu_is_not_reported_as_zero(monkeypatch):
    monkeypatch.setattr('app.gpu_metrics.shutil.which', lambda _: None)
    result = GPUMetrics().sample()
    assert result['gpu_power_watts'] is None
    assert result['gpu_utilization_percent'] is None
    assert result['gpu_source'] == 'unavailable'
