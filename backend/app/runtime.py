from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
import logging
from queue import Queue, Empty, Full
import threading
import time
from uuid import uuid4

from app.camera.camera_manager import CameraManager
from app.camera.camera_worker import CameraWorker
from app.camera.video_library import VideoLibrary
from app.camera.source_assignments import SourceAssignments
from app.camera.discovery import probe, stream_uri
from app.config.settings import CameraSourceConfig
from app.config.settings import resolve_source
from app.detection.tracker import TemporalTracker
from app.detection.yolo_detector import YoloDetector
from app.domain import Frame, Detection, utcnow
from app.events.cloud_dispatcher import CloudOutboxDispatcher
from app.events.event_manager import EventManager
from app.events.gate_logic import update_direction
from app.metrics import Metrics
from app.ocr.container_ocr import crop_identification
from app.ocr.ocr_engine import EasyOCREngine, create_ocr_engine
from app.ocr.validator import ContainerValidator
from app.ocr.job_store import OCRJobStore

log = logging.getLogger('gate')


@dataclass(frozen=True)
class OCRJob:
    origin_key: str
    gate_id: str
    track_id: int
    detection: Detection
    frame: Frame
    direction: str
    started_at: datetime
    roi: tuple
    all_detections: tuple[Detection, ...] = ()
    role: str = 'UNASSIGNED'



class GateRuntime:
    def __init__(self, settings, cameras, repository, snapshots, manager=None, detector_factory=YoloDetector,
                 ocr_factory=None, camera_sources=None):
        self.settings, self.cameras = settings, cameras
        self.manager = manager or CameraManager(cameras, settings)
        self.videos = VideoLibrary(settings.upload_directory)
        self.camera_sources = camera_sources or []
        self._sources_by_id = {source.id: source for source in self.camera_sources}
        self._default_source_by_slot = {camera.id: source.id for camera, source in zip(cameras, self.camera_sources)}
        self.source_assignments = SourceAssignments(settings.upload_directory)
        self.role_assignments = SourceAssignments(settings.upload_directory, 'camera-roles.json')
        self._registry_lock = threading.RLock()
        self._discovered = {}
        self._discovery_wake = threading.Event()
        self.discovery_status = {'state': 'WAITING', 'found': 0, 'message': 'Waiting for camera discovery'}
        for index, camera in enumerate(self.cameras):
            saved_role = self.role_assignments.get(camera.id)
            if saved_role:
                updated = type(camera).model_validate({**camera.model_dump(), **saved_role})
                self.cameras[index] = updated
                self.manager.workers[camera.id].config = updated
        self._camera_locks = {c.id: threading.RLock() for c in cameras}
        self._model_lock = threading.Lock()
        self.repository, self.snapshots = repository, snapshots
        self.detector_factory = detector_factory
        self.ocr_factory = ocr_factory or create_ocr_engine
        self.metrics = Metrics()
        self.dispatcher = CloudOutboxDispatcher(settings, self.metrics)
        self.events = EventManager(settings, repository, snapshots, dispatcher=self.dispatcher)
        self.validator = ContainerValidator()
        self.tracker = {c.id: TemporalTracker(settings.track_ttl_seconds) for c in cameras}
        self.queue = Queue(maxsize=settings.ocr_queue_size)
        self.ocr_store = OCRJobStore(settings.upload_directory / 'ocr-spool', settings.ocr_spool_max_mb,
                                     settings.snapshot_directory / 'ocr-pending')
        self.ocr_spool_error = None
        self._stop = threading.Event()
        self._accelerator = threading.RLock()
        self._state_lock = threading.Lock()
        self._threads = []
        self._recent = deque(maxlen=300)
        self._preview_detections = {}
        self._tracks = {}
        self._run_id = str(uuid4())
        self.detector = None
        self.ocr = None
        self.model_status = {'detector': 'DISABLED', 'ocr': 'DISABLED', 'error_type': None}
        self.last_processing_error = None

    def start(self):
        writer = threading.Thread(target=self._save_ocr_loop, name='gate-ocr-save', daemon=True)
        self._threads.append(writer)
        writer.start()
        for camera in self.cameras:
            saved = self.videos.info(camera.id)
            if saved and saved.get('active') and self.videos.path(saved['stored_name']).is_file():
                try:
                    self.activate_video(camera.id, saved)
                except Exception as exc:
                    log.warning('saved_video_unavailable', extra={'camera_id': camera.id, 'error_type': type(exc).__name__})
            else:
                selected = self.source_assignments.get(camera.id)
                if selected in self._sources_by_id:
                    self._install_camera_source(camera.id, selected, persist=False)
                else:
                    self.manager.workers[camera.id].start()
        if self.settings.pipeline_enabled:
            self.start_processing()
        self.dispatcher.start()
        if self.settings.camera_discovery_enabled and self.settings.deployment_mode != 'test':
            thread = threading.Thread(target=self._discovery_loop, name='gate-camera-discovery', daemon=True)
            self._threads.append(thread)
            thread.start()
        else:
            self.discovery_status = {'state': 'DISABLED', 'found': 0, 'message': 'Automatic discovery disabled'}

    def start_processing(self):
        with self._model_lock:
            if self.model_status['detector'] in ('READY', 'LOADING') and self.model_status['ocr'] in ('READY', 'LOADING'):
                return
            self.settings.pipeline_enabled = True
            self.model_status = {'detector': 'LOADING', 'ocr': 'LOADING', 'error_type': None}
            thread = threading.Thread(target=self._initialize_pipeline, name='gate-model-loader', daemon=True)
            self._threads.append(thread)
            thread.start()

    def camera_status(self):
        result = self.manager.status()
        for camera in result:
            video = self.videos.info(camera['id'])
            camera['video'] = {k: v for k, v in video.items() if k != 'stored_name'} if video else None
            selected = self.source_assignments.get(camera['id']) or self._default_source_by_slot.get(camera['id'])
            source = self._sources_by_id.get(selected)
            camera['active_source_id'] = source.id if source else None
            camera['active_source_name'] = source.name if source else None
        return result

    def available_camera_sources(self):
        with self._registry_lock:
            return [{'id': source.id, 'name': source.name,
                     'configured': bool(self._source_url(source)),
                     'discovered': source.id in self._discovered,
                     'connection_hint': self._discovered.get(source.id, {}).get('status', '')}
                    for source in self.camera_sources if source.enabled]

    def _source_url(self, source):
        return self._discovered.get(source.id, {}).get('uri', '') or resolve_source(source)

    def _discovery_loop(self):
        while not self._stop.is_set():
            self._discovery_wake.clear()
            self.discovery_status = {'state': 'SEARCHING', 'found': len(self._discovered), 'message': 'Searching connected networks for cameras…'}
            try:
                devices = probe(self._stop)
                for device in devices:
                    if self._stop.is_set():
                        return
                    try:
                        device['uri'] = stream_uri(device, self.settings.camera_onvif_username,
                                                   self.settings.camera_onvif_password.get_secret_value())
                        device['status'] = 'Stream available'
                    except Exception:
                        device['uri'] = ''
                        device['status'] = 'Found — check ONVIF credentials/settings'
                    self._register_discovered(device)
                self.discovery_status = {'state': 'COMPLETE', 'found': len(devices),
                    'message': f'{len(devices)} cameras found on this scan' if devices else 'No ONVIF cameras found. Check PoE network and camera discovery settings.'}
            except Exception as exc:
                log.warning('camera_discovery_failed', extra={'error_type': type(exc).__name__})
                self.discovery_status = {'state': 'ERROR', 'found': 0, 'message': 'Camera discovery failed; retrying automatically'}
            self._discovery_wake.wait(self.settings.camera_discovery_interval_seconds)

    def _register_discovered(self, device):
        with self._registry_lock:
            if device['id'] not in self._discovered and len(self._discovered) >= 32:
                return
            old_uri = self._discovered.get(device['id'], {}).get('uri')
            self._discovered[device['id']] = device
            if device['id'] not in self._sources_by_id:
                source = CameraSourceConfig(id=device['id'], name=device['name'], source_env='DISCOVERED_CAMERA_UNUSED')
                self.camera_sources.append(source)
                self._sources_by_id[source.id] = source
        # Restore stable device IDs after restart; never overwrite operator choices.
        assigned = [c for c in self.cameras if self.source_assignments.get(c.id) == device['id']]
        if not device.get('uri'):
            return
        for camera in assigned:
            with self._camera_locks[camera.id]:
                worker = self.manager.workers[camera.id]
                if worker.config.source_type != 'file' and old_uri != device['uri']:
                    self._install_camera_source(camera.id, device['id'], persist=False)
        if assigned:
            return
        from urllib.parse import urlsplit
        if any(urlsplit(resolve_source(source)).hostname == device['host']
               for source in self.camera_sources if source.id not in self._discovered):
            return  # Existing configured camera already owns its view.
        for camera in list(self.cameras):
            with self._camera_locks[camera.id]:
                worker = self.manager.workers[camera.id]
                if (self.source_assignments.get(camera.id) or resolve_source(camera)
                        or worker.config.source_type == 'file' or not camera.enabled):
                    continue
                self.configure_role(camera.id, 'UNASSIGNED', 'UNKNOWN')
                self._install_camera_source(camera.id, device['id'], persist=True)
                break

    def configure_role(self, camera_id, role, direction):
        with self._camera_locks[camera_id]:
            index = next(i for i, c in enumerate(self.cameras) if c.id == camera_id)
            original = self.cameras[index]
            lane = {'ENTRY': 'lane-in', 'EXIT': 'lane-out', 'UNKNOWN': 'unassigned-' + camera_id}[direction]
            values = {'role': role, 'direction': direction, 'gate_id': lane,
                      'name': f"View {index + 1} · {direction} · {role.replace('_', ' ').title()}",
                      'line_axis': None, 'ocr_roi': (0, 0, 1, 1)}
            self.role_assignments.set(camera_id, values)
            updated = original.model_copy(update=values)
            self.cameras[index] = updated
            worker = self.manager.workers[camera_id]
            worker.config = worker.config.model_copy(update=values)
            worker.source_id = str(uuid4())  # Reject queued observations using previous calibration.
            worker.take()
            self.tracker[camera_id] = TemporalTracker(self.settings.track_ttl_seconds)

    def select_camera_source(self, camera_id, source_id):
        if source_id not in self._sources_by_id or not self._sources_by_id[source_id].enabled:
            raise ValueError('Camera source is not available')
        self._install_camera_source(camera_id, source_id, persist=True)

    def _install_camera_source(self, camera_id, source_id, persist):
        with self._camera_locks[camera_id]:
            original = next(c for c in self.cameras if c.id == camera_id)
            source = self._sources_by_id[source_id]
            previous = self.manager.workers[camera_id]
            previous.stop()
            if not previous.join():
                raise RuntimeError('Previous source is still shutting down; retry shortly')
            saved = self.videos.info(camera_id)
            if saved:
                self.videos.assign(camera_id, {**saved, 'active': False})
            replacement = CameraWorker(original.model_copy(update={'source_type': 'rtsp'}), self._source_url(source), self.settings, previous.stream_factory)
            self.manager.workers[camera_id] = replacement
            self.tracker[camera_id] = TemporalTracker(self.settings.track_ttl_seconds)
            if persist:
                self.source_assignments.set(camera_id, source_id)
            replacement.start()

    def activate_video(self, camera_id, info):
        with self._camera_locks[camera_id]:
            original = next(c for c in self.cameras if c.id == camera_id)
            previous = self.manager.workers[camera_id]
            config = original.model_copy(update={'source_type': 'file', 'enabled': True, 'loop_file': False})
            replacement = CameraWorker(config, str(self.videos.path(info['stored_name'])), self.settings, previous.stream_factory)
            replacement.prepare_video()
            previous.stop()
            if not previous.join():
                raise RuntimeError('Previous source is still shutting down; retry shortly')
            self.videos.assign(camera_id, {**info, 'active': True})
            self.manager.workers[camera_id] = replacement
            self.tracker[camera_id] = TemporalTracker(self.settings.track_ttl_seconds)

    def control_video(self, camera_id, action):
        with self._camera_locks[camera_id]:
            worker = self.manager.workers[camera_id]
            if action == 'restore-camera':
                selected = self.source_assignments.get(camera_id) or self._default_source_by_slot.get(camera_id)
                if selected not in self._sources_by_id:
                    raise ValueError('Select a configured camera source')
                self._install_camera_source(camera_id, selected, persist=False)
            elif worker.config.source_type != 'file':
                raise ValueError('Add a video to this camera slot first')
            elif action == 'pause':
                worker.pause()
            elif action in ('play', 'restart'):
                if action == 'restart' or worker.status()['reason'] == 'file_completed':
                    saved = self.videos.info(camera_id)
                    if not saved:
                        raise ValueError('Restart is available for uploaded videos')
                    self.activate_video(camera_id, saved)
                self.manager.workers[camera_id].play()

    def _process_current(self, camera_id, next_at):
        # Never take a frame and then wait behind a long GPU OCR call.
        if not self._accelerator.acquire(blocking=False):
            return
        try:
            self._process_latest(camera_id, next_at)
        finally:
            self._accelerator.release()

    def _process_latest(self, camera_id, next_at):
        with self._camera_locks[camera_id]:
            camera = self.manager.workers[camera_id].config
            now = time.monotonic()
            if now < next_at[camera_id]:
                return
            worker = self.manager.workers[camera_id]
            if worker.status()['playback_status'] == 'PAUSED':
                return
            frame = worker.take()
            if frame is None:
                return
            if frame.sequence % (camera.frame_skip + 1):
                self.metrics.increment('frames_skipped')
                return
            next_at[camera_id] = now + 1 / (camera.inference_fps or self.settings.inference_fps)
            self.process_frame(camera, frame)

    def _initialize_pipeline(self):
        try:
            self.detector = self.detector_factory(self.settings)
            self.model_status['detector'] = 'READY'
            self.ocr = self.ocr_factory(self.settings)
            self.model_status['ocr'] = 'READY'
            if self._stop.is_set():
                return
            log.info('models_loaded')
            for name, target in [('gate-inference', self._inference_loop), ('gate-ocr', self._ocr_loop)]:
                thread = threading.Thread(target=target, name=name, daemon=True)
                self._threads.append(thread)
                thread.start()
        except Exception as exc:
            if self.detector is None:
                self.model_status['detector'] = 'ERROR'
            self.model_status['ocr'] = 'ERROR'
            self.model_status['error_type'] = type(exc).__name__
            log.error('model_load_failed', extra={'error_type': type(exc).__name__})

    def stop(self):
        self._stop.set()
        self._discovery_wake.set()
        self.dispatcher.stop()
        cameras_stopped = self.manager.stop()
        for thread in list(self._threads):
            thread.join(timeout=10)
        clean = cameras_stopped and not any(t.is_alive() for t in self._threads)
        if not clean:
            log.error('worker_shutdown_timeout')
        return clean

    def recent_detections(self, limit=50, camera_id=None):
        with self._state_lock:
            values = list(reversed(self._recent))
        return [d for d in values if camera_id is None or d['camera_id'] == camera_id][:limit]

    def _inference_loop(self):
        next_at = {c.id: 0. for c in self.cameras}
        while not self._stop.is_set():
            for camera in self.cameras:
                if self._stop.is_set():
                    break
                try:
                    self._process_current(camera.id, next_at)
                except Exception as exc:
                    self.metrics.increment('inference_errors')
                    self.last_processing_error = type(exc).__name__
                    log.error('inference_failed', extra={'camera_id': camera.id, 'error_type': type(exc).__name__})
            self._stop.wait(.005)

    def process_frame(self, camera, frame):
        with self._accelerator:
            started = time.monotonic()
            detections = self.detector.detect(frame)
            self.metrics.latency('inference', time.monotonic() - started)
        self.metrics.inference(camera.id)
        self.metrics.increment('inferences')
        tracks = self.tracker[camera.id].update(detections, frame.timestamp)
        with self._state_lock:
            self._preview_detections[camera.id] = (frame.source_id, frame.timestamp, tuple(detections))
            for detection in detections:
                self._recent.append({**asdict(detection), 'frame_timestamp': detection.frame_timestamp.isoformat() + 'Z'})
            cutoff = self.settings.track_ttl_seconds * 4
            self._tracks = {key: value for key, value in self._tracks.items()
                            if value['pending'] or (frame.timestamp - value['seen']).total_seconds() <= cutoff}
        for track in tracks:
            direction = update_direction(camera, track, frame.image.shape)
            origin = f'{self._run_id}:{camera.id}:{frame.source_id[:12]}:{track.id}'
            with self._state_lock:
                state = self._tracks.setdefault(origin, {'pending': False, 'done': False, 'last_ocr': 0., 'seen': frame.timestamp, 'samples': 0})
                state['seen'] = frame.timestamp
                if state['pending'] or state['done'] or time.monotonic() - state['last_ocr'] < self.settings.ocr_interval_seconds:
                    continue
                if state.get('samples', 0) >= self.settings.ocr_max_attempts:
                    continue
                if track.hits < self.settings.min_track_hits:
                    continue
                if track.detection.class_name not in self.settings.ocr_classes:
                    continue
                if camera.line_axis and direction == 'UNKNOWN':
                    continue
                job = OCRJob(origin, camera.gate_id, track.id, track.detection, frame, direction, track.first_seen, camera.ocr_roi, tuple(detections), camera.role)
                try:
                    self.queue.put_nowait(job)
                    state['pending'] = True
                    state['samples'] = state.get('samples', 0) + 1
                    state['last_ocr'] = time.monotonic()
                except Full:
                    self.metrics.increment('ocr_queue_dropped')

    def _save_ocr_loop(self):
        """Disk I/O is isolated from capture and YOLO. Retry the current write."""
        while not self._stop.is_set():
            try:
                job = self.queue.get(timeout=.2)
            except Empty:
                continue
            try:
                crop = crop_identification(job.frame.image, job.detection.bbox, job.roi)
                while not self._stop.is_set():
                    try:
                        self.ocr_store.save(job, crop)
                        self.ocr_spool_error = None
                        self.metrics.increment('ocr_jobs_saved')
                        with self._state_lock:
                            if job.origin_key in self._tracks:
                                self._tracks[job.origin_key]['pending'] = False
                        break
                    except Exception as exc:
                        self.ocr_spool_error = type(exc).__name__
                        self.metrics.increment('ocr_spool_write_errors')
                        self._stop.wait(1)
            except Exception as exc:
                self.ocr_spool_error = type(exc).__name__
                self.metrics.increment('ocr_spool_write_errors')
                with self._state_lock:
                    if job.origin_key in self._tracks:
                        self._tracks[job.origin_key]['pending'] = False
            finally:
                self.queue.task_done()

    def _process_saved_ocr(self, job_id):
        job, crop, validation = self.ocr_store.load(job_id)
        if validation is None:
            with self._accelerator if self.settings.ocr_gpu else nullcontext():
                started = time.monotonic()
                read = self.ocr.read(crop)
                self.metrics.latency('ocr', time.monotonic() - started)
            validation = self.validator.validate(read)
            self.ocr_store.save_validation(job_id, validation)
        # Recover confirmation votes from disk, including after a restart.
        # Saved jobs retain historical source/lane/role; never relabel them to
        # the currently selected camera or attach current-time companion images.
        self.events.evidence[job.origin_key] = self.ocr_store.evidence(job.origin_key, self.settings.ocr_max_attempts)
        try:
            result = self.events.observe(job, validation)
            self.ocr_store.complete(job_id, result[0] if result else None)
        finally:
            self.events.evidence.pop(job.origin_key, None)
        with self._state_lock:
            state = self._tracks.get(job.origin_key)
            if state:
                state['pending'] = False
                state['done'] = bool(result)
        if result:
            self.metrics.increment('events_created' if result[1] else 'evidence_associated')
            self.metrics.latency('event', (utcnow() - job.started_at).total_seconds())

    def _ocr_loop(self):
        while not self._stop.is_set():
            job_id = None
            try:
                job_id = self.ocr_store.next_id()
                if job_id is None:
                    self._stop.wait(.2)
                    continue
                self._process_saved_ocr(job_id)
                self.last_processing_error = None
            except Exception as exc:
                self.last_processing_error = type(exc).__name__
                self.metrics.increment('ocr_job_errors')
                log.error('saved_ocr_failed', extra={'error_type': type(exc).__name__})
                if job_id is not None:
                    try:
                        self.ocr_store.retry(job_id, type(exc).__name__)
                    except Exception:
                        self.ocr_spool_error = 'RetryWriteFailed'
                self._stop.wait(.2)
