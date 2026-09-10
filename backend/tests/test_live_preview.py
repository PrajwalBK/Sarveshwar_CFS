import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from app.main import create_app


def test_live_preview_latest_frame_and_origin(settings, repository, camera):
    with TestClient(create_app(settings, repository.db, [camera], camera_sources=[])) as client:
        cam = client.app.state.runtime.manager.workers[camera.id]
        with client.websocket_connect('/api/cameras/camera-1/live', headers={'origin': 'http://testserver'}) as ws:
            ws.send_text('next')
            assert ws.receive_text() == 'waiting'
            cam._publish(np.zeros((100, 150, 3), dtype=np.uint8))
            ws.send_text('next')
            first = ws.receive_bytes()
            assert first.startswith(b'\xff\xd8')
            cam._publish(np.full((100, 150, 3), 255, dtype=np.uint8))
            ws.send_text('next')
            assert ws.receive_bytes() != first
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/api/cameras/camera-1/live', headers={'origin': 'https://untrusted.example'}):
                pass


def test_busy_accelerator_does_not_consume_latest_frame(settings, repository, camera):
    import threading
    with TestClient(create_app(settings, repository.db, [camera], camera_sources=[])) as client:
        runtime = client.app.state.runtime
        cam = runtime.manager.workers[camera.id]
        cam._publish(np.zeros((100, 150, 3), dtype=np.uint8))
        held, release = threading.Event(), threading.Event()
        def hold():
            with runtime._accelerator:
                held.set()
                release.wait(3)
        thread = threading.Thread(target=hold)
        thread.start()
        try:
            assert held.wait(1)
            runtime._process_current(camera.id, {camera.id: 0})
            assert cam.take() is not None
        finally:
            release.set()
            thread.join()


def test_eight_live_connections_leave_health_available(settings, repository, camera):
    from contextlib import ExitStack
    configs = [camera.model_copy(update={'id': f'camera-{i}'}) for i in range(8)]
    with TestClient(create_app(settings, repository.db, configs, camera_sources=[])) as client:
        with ExitStack() as stack:
            connections = []
            for config in configs:
                client.app.state.runtime.manager.workers[config.id]._publish(np.zeros((40, 60, 3), dtype=np.uint8))
                connections.append(stack.enter_context(client.websocket_connect(
                    f'/api/cameras/{config.id}/live', headers={'origin': 'http://testserver'})))
            for socket in connections:
                socket.send_text('next')
                assert socket.receive_bytes().startswith(b'\xff\xd8')
            assert client.get('/api/health').status_code == 200
