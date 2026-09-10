import pytest
from fastapi.testclient import TestClient
from app.camera.discovery import parse_probe, same_host_url, xml
from app.camera.camera_worker import CameraWorker
from app.config.settings import CameraConfig
from app.main import create_app


def test_probe_stable_id_and_endpoint_validation():
    data = b'''<Envelope><ProbeMatch><EndpointReference><Address>urn:uuid:camera-one</Address></EndpointReference><XAddrs>http://192.168.1.8/onvif/device_service</XAddrs></ProbeMatch></Envelope>'''
    first = parse_probe(data, '192.168.1.8')
    assert len(first) == 1 and first[0]['name'] == 'Camera 192.168.1.8'
    assert parse_probe(data, '192.168.1.9') == []
    assert parse_probe(data, '127.0.0.1') == []
    moved = data.replace(b'192.168.1.8', b'192.168.1.9')
    assert parse_probe(moved, '192.168.1.9')[0]['id'] == first[0]['id']
    with pytest.raises(ValueError):
        xml(b'<!DOCTYPE x [<!ENTITY e "bad">]><x/>')
    with pytest.raises(ValueError):
        same_host_url('http://user:password@192.168.1.8/', '192.168.1.8')


def test_auto_assignment_roles_restart_and_redaction(settings, repository, monkeypatch):
    # Do not contact real RTSP devices in this test.
    monkeypatch.setattr(CameraWorker, 'start', lambda self: None)
    configs = [CameraConfig(id=f'slot-{i}', name=f'Slot {i}', gate_id='lane-in',
                            source_env=f'TEST_BLANK_{i}', direction='ENTRY') for i in range(3)]
    device = {'id': 'onvif-one', 'name': 'Camera 192.168.1.8', 'host': '192.168.1.8',
              'uri': 'rtsp://operator:secret@192.168.1.8/stream', 'status': 'Stream available'}
    with TestClient(create_app(settings, repository.db, configs, camera_sources=[])) as client:
        runtime = client.app.state.runtime
        runtime.source_assignments.set('slot-0', 'operator-choice')
        runtime._register_discovered(device)
        slots = client.get('/api/cameras').json()
        assert runtime.source_assignments.get('slot-0') == 'operator-choice'
        assert slots[1]['active_source_id'] == 'onvif-one'
        assert slots[1]['direction'] == 'UNKNOWN'
        assert slots[1]['role'] == 'UNASSIGNED'
        before = runtime.manager.workers['slot-1'].source_id
        assert client.post('/api/cameras/slot-1/role', json={'role': 'LEFT', 'direction': 'EXIT'}).status_code == 200
        assert runtime.manager.workers['slot-1'].source_id != before
        runtime._register_discovered(device)
        assert runtime.manager.workers['slot-1'].config.direction == 'EXIT'
        assert runtime.source_assignments.get('slot-2') is None
        assert 'secret' not in client.get('/api/cameras/sources/available').text
        assert 'secret' not in client.get('/api/cameras').text
        assert client.post('/api/cameras/slot-1/role', json={'role': 'invalid', 'direction': 'EXIT'}).status_code == 422
        assert client.get('/api/cameras/discovery/status').json()['state'] == 'DISABLED'
    with TestClient(create_app(settings, repository.db, configs, camera_sources=[])) as client:
        runtime = client.app.state.runtime
        runtime._register_discovered(device)
        slot = client.get('/api/cameras').json()[1]
        assert slot['role'] == 'LEFT' and slot['direction'] == 'EXIT'
        assert slot['active_source_id'] == 'onvif-one'


def test_found_camera_without_stream_is_listed_without_replacing_video(settings, repository, camera, monkeypatch):
    monkeypatch.setattr(CameraWorker, 'start', lambda self: None)
    with TestClient(create_app(settings, repository.db, [camera], camera_sources=[])) as client:
        runtime = client.app.state.runtime
        before = runtime.manager.workers[camera.id]
        before.config = before.config.model_copy(update={'source_type': 'file'})
        runtime._register_discovered({'id': 'onvif-auth', 'name': 'Camera', 'host': '192.168.1.10',
                                      'uri': '', 'status': 'Needs credentials'})
        assert runtime.manager.workers[camera.id] is before
        assert client.get('/api/cameras/sources/available').json()[0]['configured'] is False
        runtime._register_discovered({'id': 'onvif-ready', 'name': 'Camera 2', 'host': '192.168.1.11',
                                      'uri': 'rtsp://192.168.1.11/stream', 'status': 'Stream available'})
        assert runtime.manager.workers[camera.id] is before
        assert runtime.source_assignments.get(camera.id) is None


def test_media_service_negotiation_and_credentials(monkeypatch):
    from io import BytesIO
    from app.camera import discovery
    requests = []
    responses = iter([
        b'<Envelope><Media><XAddr>http://192.168.1.8/media</XAddr></Media></Envelope>',
        b'<Envelope><Profiles token="profile&amp;one"/></Envelope>',
        b'<Envelope><Uri>rtsp://192.168.1.8:554/live</Uri></Envelope>',
    ])

    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            assert timeout == 3
            return BytesIO(next(responses))

    monkeypatch.setattr(discovery, 'build_opener', lambda *args: Opener())
    device = {'host': '192.168.1.8', 'endpoint': 'http://192.168.1.8/device'}
    assert discovery.stream_uri(device, 'user@site', 'p:word') == 'rtsp://user%40site:p%3Aword@192.168.1.8:554/live'
    assert len(requests) == 3
    assert b'profile&amp;one' in requests[2].data
    assert b'PasswordDigest' in requests[0].data
    assert b'p:word' not in requests[0].data

    responses = iter([b'<Envelope><Media><XAddr>http://192.168.1.9/media</XAddr></Media></Envelope>'])
    with pytest.raises(ValueError):
        discovery.stream_uri(device)
