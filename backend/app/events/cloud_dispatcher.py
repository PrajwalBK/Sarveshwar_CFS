"""
Prosper Smart Yard Cloud Outbox Dispatcher
Asynchronously synchronizes confirmed GateVision events and evidence images
to the remote CFS Smart Yard API (cfsapi.prosperassettracking.com / syapi.prosperassettracking.com).
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import io
import logging
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

import cv2
import numpy as np
import requests

log = logging.getLogger('gate.cloud')


def _iso_utc(dt: Any = None) -> str:
    if dt is None:
        d = datetime.now(timezone.utc)
    elif isinstance(dt, (int, float)):
        try:
            d = datetime.fromtimestamp(dt, tz=timezone.utc)
        except Exception:
            d = datetime.now(timezone.utc)
    elif isinstance(dt, str):
        if dt.endswith('Z') or '+' in dt or '-' in dt[10:]:
            return dt
        return f"{dt}Z"
    elif isinstance(dt, datetime):
        if dt.tzinfo is None:
            d = dt.replace(tzinfo=timezone.utc)
        else:
            d = dt.astimezone(timezone.utc)
    else:
        d = datetime.now(timezone.utc)
    return d.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def map_camera_to_image_type(camera_id: str | None) -> str:
    """Map camera ID or position to CFS imageType: FRONT, REAR, LEFT, RIGHT."""
    cid = (camera_id or '').lower()
    if 'rear' in cid or cid.endswith('-4') or cid.endswith('_4') or cid.endswith('cam-4'):
        return 'REAR'
    if 'left' in cid or cid.endswith('-2') or cid.endswith('_2') or cid.endswith('cam-2'):
        return 'LEFT'
    if 'right' in cid or cid.endswith('-3') or cid.endswith('_3') or cid.endswith('cam-3'):
        return 'RIGHT'
    return 'FRONT'


def to_base64_data_uri(image: Any, quality: int = 85) -> Optional[str]:
    """Convert numpy array, raw bytes, or file path to data:image/jpeg;base64 URI."""
    if image is None:
        return None
    if isinstance(image, str):
        if image.startswith('data:image'):
            return image
        p = Path(image)
        if p.is_file():
            b64 = base64.b64encode(p.read_bytes()).decode('utf-8')
            return f'data:image/jpeg;base64,{b64}'
        return None
    if isinstance(image, Path):
        if image.is_file():
            b64 = base64.b64encode(image.read_bytes()).decode('utf-8')
            return f'data:image/jpeg;base64,{b64}'
        return None
    if isinstance(image, bytes):
        b64 = base64.b64encode(image).decode('utf-8')
        return f'data:image/jpeg;base64,{b64}'
    if isinstance(image, np.ndarray):
        ok, buf = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        return f'data:image/jpeg;base64,{b64}'
    return None


def build_cfs_capture_payload_from_obs(
    event_id: str,
    obs: Any,
    device_id: str,
    additional_frames: Optional[Dict[str, Any]] = None,
    primary_image_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Build GateEventCaptureRequest from an in-flight Observation."""
    # Determine event type
    direction = getattr(obs, 'direction', '')
    gate_id = getattr(obs, 'gate_id', '')
    if direction == 'ENTRY':
        event_type = 'GATE_IN'
    elif direction == 'EXIT':
        event_type = 'GATE_OUT'
    elif 'out' in str(gate_id).lower():
        event_type = 'GATE_OUT'
    else:
        event_type = 'GATE_IN'

    timestamp_iso = _iso_utc(getattr(obs.detection, 'frame_timestamp', None) if getattr(obs, 'detection', None) else None)
    visit_id = f"VISIT-{event_id.replace('-', '')[:8].upper()}-{int(time.time()) % 10000}"

    container_num = obs.ocr.normalized_text if (obs.ocr and obs.ocr.normalized_text) else None
    confidence = float(obs.ocr.confidence) if (obs.ocr and obs.ocr.confidence is not None) else None

    # Collect images
    images: List[Dict[str, Any]] = []
    seen_cameras = set()

    # 1. Primary camera frame
    primary_cam = obs.detection.camera_id if getattr(obs, 'detection', None) else 'gate-in-1'
    primary_img = primary_image_bytes if primary_image_bytes is not None else getattr(obs, 'image', None)
    b64_primary = to_base64_data_uri(primary_img)
    if b64_primary:
        images.append({
            'imageType': map_camera_to_image_type(primary_cam),
            'cameraId': primary_cam,
            'capturedAt': timestamp_iso,
            'image': b64_primary,
        })
        seen_cameras.add(primary_cam)

    # 2. Additional lane camera frames
    if additional_frames:
        for cam_id, frame_data in additional_frames.items():
            if cam_id in seen_cameras:
                continue
            b64_frame = to_base64_data_uri(frame_data)
            if b64_frame:
                images.append({
                    'imageType': map_camera_to_image_type(cam_id),
                    'cameraId': cam_id,
                    'capturedAt': timestamp_iso,
                    'image': b64_frame,
                })
                seen_cameras.add(cam_id)

    payload = {
        'visitId': visit_id,
        'eventType': event_type,
        'deviceId': device_id,
        'capturedAt': timestamp_iso,
        'container': {
            'containerNumber': container_num,
            'containerNumberConfidence': confidence,
            'size': '40FT' if container_num else None,
        } if container_num else None,
        'truck': {
            'truckNumber': None,
        },
        'driver': {
            'driverName': None,
            'driverId': None,
        },
        'images': images,
    }
    return payload


def build_cfs_capture_payload_from_record(
    event_record: Dict[str, Any],
    snapshots_manager: Any,
    device_id: str,
) -> Dict[str, Any]:
    """Build GateEventCaptureRequest from a saved database event record and snapshots."""
    event_id = event_record['id']
    raw_event_type = event_record.get('event_type')
    gate_id = event_record.get('gate_id', '')
    if raw_event_type == 'ENTRY':
        event_type = 'GATE_IN'
    elif raw_event_type == 'EXIT':
        event_type = 'GATE_OUT'
    elif 'out' in str(gate_id).lower():
        event_type = 'GATE_OUT'
    else:
        event_type = 'GATE_IN'

    timestamp_iso = _iso_utc(event_record.get('timestamp'))
    visit_id = f"VISIT-{event_id.replace('-', '')[:8].upper()}"

    container_num = event_record.get('container_number')
    confidence = float(event_record.get('confidence')) if event_record.get('confidence') is not None else None

    images: List[Dict[str, Any]] = []
    seen_types = set()

    for snap in event_record.get('snapshots', []):
        rel_path = snap.get('image_path')
        if not rel_path:
            continue
        try:
            full_path = snapshots_manager.path(rel_path)
            b64_uri = to_base64_data_uri(full_path)
        except Exception:
            b64_uri = None

        if b64_uri:
            cam_id = snap.get('camera_id') or 'CAM'
            img_type = map_camera_to_image_type(cam_id)
            # If we have multiple for the same position, still include or disambiguate
            images.append({
                'imageType': img_type,
                'cameraId': cam_id,
                'capturedAt': _iso_utc(snap.get('timestamp', timestamp_iso)),
                'image': b64_uri,
            })
            seen_types.add(img_type)

    payload = {
        'visitId': visit_id,
        'eventType': event_type,
        'deviceId': device_id,
        'capturedAt': timestamp_iso,
        'container': {
            'containerNumber': container_num,
            'containerNumberConfidence': confidence,
            'size': '40FT' if container_num else None,
        } if container_num else None,
        'truck': {
            'truckNumber': None,
        },
        'driver': {
            'driverName': None,
            'driverId': None,
        },
        'images': images,
    }
    return payload


class ProsperSession:
    """Manages authentication tokens and HTTP requests to Prosper Smart Yard / CFS API."""

    def __init__(self, settings):
        self.settings = settings
        self.base_url = settings.prosper_api_base_url.rstrip('/')
        self.site_id = (settings.prosper_site_id or '').strip()
        self.session = requests.Session()
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._lock = threading.Lock()
        self.last_image_metadata_id: Optional[str] = None

    def get_auth_headers(self, force_refresh: bool = False) -> Dict[str, str]:
        """Return Authorization or API key headers, auto-refreshing JWT when necessary."""
        if self.settings.prosper_bearer_token and not force_refresh:
            return {'Authorization': f'Bearer {self.settings.prosper_bearer_token.get_secret_value()}'}
        if self.settings.prosper_api_key and not force_refresh:
            return {'x-api-key': self.settings.prosper_api_key.get_secret_value()}

        # Auto-login credential flow
        with self._lock:
            now = time.monotonic()
            if not force_refresh and self._token and now < self._token_expires_at - 30:
                return {'Authorization': f'Bearer {self._token}'}

            username = getattr(self.settings, 'prosper_username', None) or self.settings.prosper_email
            password = self.settings.prosper_password.get_secret_value() if self.settings.prosper_password else ''

            if not (username and password) and not self.settings.prosper_site_code:
                # If static bearer token was configured, fallback to it even if force_refresh was requested
                if self.settings.prosper_bearer_token:
                    return {'Authorization': f'Bearer {self.settings.prosper_bearer_token.get_secret_value()}'}
                return {}

            try:
                login_url = f'{self.base_url}/api/auth/login'
                payload = {
                    'UserName': username,
                    'Password': password,
                }
                if self.settings.prosper_site_code:
                    payload['siteCode'] = self.settings.prosper_site_code

                res = self.session.post(login_url, json=payload, timeout=12)
                if res.status_code not in (200, 201):
                    # Fallback to legacy email / siteCode format
                    legacy_payload = {
                        'siteCode': self.settings.prosper_site_code or '',
                        'email': self.settings.prosper_email or username,
                        'password': password,
                    }
                    res = self.session.post(login_url, json=legacy_payload, timeout=12)

                if res.status_code in (200, 201):
                    data = res.json()
                    self._token = data.get('accessToken') or data.get('token')
                    expires_in = data.get('expiresIn', 86400)
                    self._token_expires_at = time.monotonic() + float(expires_in)
                    log.info('prosper_auth_token_refreshed', extra={'username': username})
                    return {'Authorization': f'Bearer {self._token}'}
                log.warning('prosper_auth_login_failed', extra={'status_code': res.status_code, 'body': res.text[:200]})
            except Exception as e:
                log.warning('prosper_auth_login_error', extra={'error': str(e)})

        # Final fallback to static token if set
        if self.settings.prosper_bearer_token:
            return {'Authorization': f'Bearer {self.settings.prosper_bearer_token.get_secret_value()}'}
        return {}

    def capture_gate_event(self, payload: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        """Send GateEventCaptureRequest to POST /api/gate-events/capture."""
        url = f'{self.base_url}/api/gate-events/capture'
        headers = self.get_auth_headers()
        headers['Content-Type'] = 'application/json'

        try:
            res = self.session.post(url, json=payload, headers=headers, timeout=30)
            if res.status_code in (200, 201):
                resp_json = res.json() if res.content else {}
                log.info('cfs_gate_event_captured', extra={'visitId': payload.get('visitId'), 'cfs_id': resp_json.get('id')})
                return True, resp_json

            # Re-authenticate on 401 Unauthorized once
            if res.status_code == 401:
                log.info('cfs_capture_token_expired_retrying_login')
                headers = self.get_auth_headers(force_refresh=True)
                headers['Content-Type'] = 'application/json'
                retry_res = self.session.post(url, json=payload, headers=headers, timeout=30)
                if retry_res.status_code in (200, 201):
                    resp_json = retry_res.json() if retry_res.content else {}
                    log.info('cfs_gate_event_captured_after_reauth', extra={'cfs_id': resp_json.get('id')})
                    return True, resp_json
                return False, {'status_code': retry_res.status_code, 'error': retry_res.text[:300]}

            log.warning('cfs_gate_event_rejected', extra={'status_code': res.status_code, 'body': res.text[:300]})
            return False, {'status_code': res.status_code, 'error': res.text[:300]}
        except Exception as exc:
            log.warning('cfs_capture_exception', extra={'error': str(exc)})
            return False, {'error': str(exc)}

    def upload_image(self, image_data: bytes | Any, trailer_number: str = '', camera_id: str = '',
                     device_id: str = '') -> Optional[str]:
        """Legacy upload snapshot to Prosper site endpoint and return public/stored URL."""
        if not self.site_id:
            return None

        if not isinstance(image_data, bytes):
            ok, buf = cv2.imencode('.jpg', image_data, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return None
            image_data = buf.tobytes()

        headers = self.get_auth_headers()
        base64_str = base64.b64encode(image_data).decode('utf-8')
        base64_uri = f'data:image/jpeg;base64,{base64_str}'

        # 1. Try S3 Base64 upload endpoint
        s3_url = f'{self.base_url}/api/sites/{self.site_id}/images/trailers/upload-s3'
        s3_payload = {
            'trailerNumber': trailer_number or 'UNKNOWN',
            'scac': trailer_number[:4] if len(trailer_number) >= 4 and trailer_number[:4].isalpha() else 'UNKN',
            'deviceId': device_id,
            'cameraId': camera_id,
            'imagePath': base64_uri,
        }
        try:
            res = self.session.post(s3_url, json=s3_payload, headers={**headers, 'Content-Type': 'application/json'}, timeout=15)
            if res.status_code in (200, 201):
                data = res.json() if res.content else {}
                nested = data.get('data') if isinstance(data.get('data'), dict) else {}
                img_id = data.get('id') or nested.get('id') or nested.get('imageMetadataId') or data.get('imageMetadataId')
                img_url = nested.get('fileUrl') or nested.get('imagePath') or data.get('fileUrl') or data.get('imagePath') or data.get('url')
                self.last_image_metadata_id = img_id
                log.info('prosper_image_s3_uploaded', extra={'img_id': img_id, 'url': img_url})
                return img_url
        except Exception as e:
            log.warning('prosper_image_s3_upload_error', extra={'error': str(e)})

        # 2. Fallback to multipart upload
        url = f'{self.base_url}/api/sites/{self.site_id}/images/trailers'
        files = {'file': ('evidence.jpg', io.BytesIO(image_data), 'image/jpeg')}
        try:
            res = self.session.post(url, files=files, headers=headers, timeout=15)
            if res.status_code in (200, 201):
                data = res.json() if res.content else {}
                img_id = data.get('id') or data.get('imageMetadataId')
                img_url = data.get('url') or data.get('imageUrl') or data.get('fileUrl')
                self.last_image_metadata_id = img_id
                log.info('prosper_image_uploaded', extra={'url': img_url})
                return img_url
            log.warning('prosper_image_upload_failed', extra={'status_code': res.status_code})
        except Exception as e:
            log.warning('prosper_image_upload_error', extra={'error': str(e)})

        return None

    def post_gate_event(self, payload: Dict[str, Any]) -> bool:
        """Legacy send CreateGateEventRequest to Prosper site endpoint."""
        if not self.site_id:
            log.warning('prosper_sync_skipped_missing_site_id')
            return False

        url = f'{self.base_url}/api/sites/{self.site_id}/gate-events'
        headers = self.get_auth_headers()
        headers['Content-Type'] = 'application/json'

        res = self.session.post(url, json=payload, headers=headers, timeout=15)
        if res.status_code in (200, 201):
            log.info('prosper_gate_event_synced', extra={'event_id': payload.get('id')})
            return True
        log.warning('prosper_gate_event_rejected', extra={'status_code': res.status_code, 'body': res.text[:200]})
        return False


class CloudOutboxDispatcher:
    """
    Dedicated non-blocking worker thread that serializes confirmed gate events,
    encodes snapshot evidence images, and dispatches gate events to CFS Smart Yard API.
    """

    def __init__(self, settings, metrics=None):
        self.settings = settings
        self.metrics = metrics
        self.enabled = bool(settings.prosper_enabled)
        self.session = ProsperSession(settings) if self.enabled else None
        self.queue: Queue = Queue(maxsize=settings.prosper_queue_size)
        self._stop = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Deterministic device UUID fallback
        self.device_uuid = (
            settings.prosper_device_uuid
            or str(uuid.uuid5(uuid.NAMESPACE_DNS, 'gatevision.edge.local'))
        )

    def start(self):
        if not self.enabled:
            return
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, name='prosper-dispatcher', daemon=True)
        self._worker_thread.start()
        log.info('prosper_cloud_dispatcher_started')

    def stop(self, timeout: float = 2.0):
        if not self.enabled:
            return
        self._stop.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        log.info('prosper_cloud_dispatcher_stopped')

    def enqueue(
        self,
        obs: Any,
        event_id: str,
        image_bytes: Optional[bytes] = None,
        additional_frames: Optional[Dict[str, Any]] = None,
    ):
        """Enqueue confirmed gate event for background dispatch."""
        if not self.enabled:
            return
        if not obs.confirmed:
            return

        item = {
            'event_id': event_id,
            'obs': obs,
            'image_bytes': image_bytes if image_bytes is not None else getattr(obs, 'image', None),
            'additional_frames': additional_frames,
            'enqueued_at': time.monotonic(),
            'retries': 0,
        }

        try:
            self.queue.put_nowait(item)
            if self.metrics:
                self.metrics.increment('prosper_events_queued')
        except Full:
            log.warning('prosper_cloud_queue_full_event_dropped', extra={'event_id': event_id})
            if self.metrics:
                self.metrics.increment('prosper_queue_dropped')

    def dispatch_event_record(self, event_record: Dict[str, Any], snapshots_manager: Any) -> tuple[bool, Dict[str, Any]]:
        """Synchronously dispatch an existing saved event to the CFS API (for manual sync API)."""
        if not self.enabled or not self.session:
            return False, {'error': 'Prosper cloud sync is disabled'}
        payload = build_cfs_capture_payload_from_record(event_record, snapshots_manager, device_id=self.device_uuid)
        return self.session.capture_gate_event(payload)

    def _run_loop(self):
        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=0.5)
            except Empty:
                continue

            event_id = item['event_id']
            obs = item['obs']
            img = item['image_bytes']
            additional_frames = item.get('additional_frames')

            success = False
            try:
                # If site_id is configured, use legacy site endpoint flow
                if self.settings.prosper_site_id:
                    image_url = None
                    cntr = obs.ocr.normalized_text if obs.ocr else ''
                    if img is not None:
                        image_url = self.session.upload_image(
                            img,
                            trailer_number=cntr,
                            camera_id=obs.detection.camera_id if getattr(obs, 'detection', None) else '',
                            device_id=self.device_uuid,
                        )
                    img_id = getattr(self.session, 'last_image_metadata_id', None)

                    payload = {
                        'id': event_id,
                        'gateId': obs.gate_id,
                        'timestamp': _iso_utc(obs.detection.frame_timestamp if getattr(obs, 'detection', None) else None),
                        'containerNumber': cntr or None,
                        'trailerNumber': cntr or None,
                        'eventType': obs.direction if obs.direction in ('ENTRY', 'EXIT') else 'ENTRY',
                        'deviceId': self.device_uuid,
                        'confidence': float(obs.detection.confidence if getattr(obs, 'detection', None) else 0.0),
                    }
                    if img_id:
                        payload['imageMetadataId'] = img_id
                    if image_url:
                        payload['imageUrl'] = image_url

                    success = self.session.post_gate_event(payload)
                else:
                    # CFS Smart Yard capture endpoint (/api/gate-events/capture)
                    capture_payload = build_cfs_capture_payload_from_obs(
                        event_id=event_id,
                        obs=obs,
                        device_id=self.device_uuid,
                        additional_frames=additional_frames,
                        primary_image_bytes=img if isinstance(img, bytes) else None,
                    )
                    success, _ = self.session.capture_gate_event(capture_payload)

            except Exception as exc:
                log.warning('prosper_dispatch_exception', extra={'event_id': event_id, 'error': str(exc)})
                success = False

            if success:
                if self.metrics:
                    self.metrics.increment('prosper_events_synced')
                    self.metrics.latency('prosper_sync', time.monotonic() - item['enqueued_at'])
                self.queue.task_done()
            else:
                item['retries'] += 1
                if item['retries'] < self.settings.prosper_max_retries:
                    backoff = min(2.0 ** item['retries'], 30.0)
                    self._stop.wait(backoff)
                    try:
                        self.queue.put_nowait(item)
                    except Full:
                        log.error('prosper_retry_dropped_queue_full', extra={'event_id': event_id})
                else:
                    log.error('prosper_event_sync_exhausted', extra={'event_id': event_id})
                    if self.metrics:
                        self.metrics.increment('prosper_sync_errors')
                self.queue.task_done()
