from collections import deque
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
        self._stop = threading.Event()
        self._accelerator = threading.Lock()
        self._state_lock = threading.Lock()
        self._threads = []
        self._recent = deque(maxlen=300)
        self._tracks = {}
        self._run_id = str(uuid4())
        self.detector = None
        self.ocr = None
        self.model_status = {'detector': 'DISABLED', 'ocr': 'DISABLED', 'error_type': None}
        self.last_processing_error = None

    def start(self):
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
        return [{'id': source.id, 'name': source.name, 'configured': bool(resolve_source(source))}
                for source in self.camera_sources if source.enabled]

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
            replacement = CameraWorker(original, resolve_source(source), self.settings, previous.stream_factory)
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
            for detection in detections:
                self._recent.append({**asdict(detection), 'frame_timestamp': detection.frame_timestamp.isoformat() + 'Z'})
            cutoff = self.settings.track_ttl_seconds * 4
            self._tracks = {key: value for key, value in self._tracks.items()
                            if value['pending'] or (frame.timestamp - value['seen']).total_seconds() <= cutoff}
        for track in tracks:
            direction = update_direction(camera, track, frame.image.shape)
            origin = f'{self._run_id}:{camera.id}:{frame.source_id[:12]}:{track.id}'
            with self._state_lock:
                state = self._tracks.setdefault(origin, {'pending': False, 'done': False, 'last_ocr': 0., 'seen': frame.timestamp})
                state['seen'] = frame.timestamp
                if state['pending'] or state['done'] or time.monotonic() - state['last_ocr'] < self.settings.ocr_interval_seconds:
                    continue
                if track.hits < self.settings.min_track_hits:
                    continue
                if track.detection.class_name not in self.settings.ocr_classes:
                    continue
                if camera.line_axis and direction == 'UNKNOWN':
                    continue
                job = OCRJob(origin, camera.gate_id, track.id, track.detection, frame, direction, track.first_seen, camera.ocr_roi, tuple(detections))
                try:
                    self.queue.put_nowait(job)
                    state['pending'] = True
                    state['last_ocr'] = time.monotonic()
                except Full:
                    self.metrics.increment('ocr_queue_dropped')

    def _ocr_loop(self):
        while not self._stop.is_set():
            try:
                job = self.queue.get(timeout=.2)
            except Empty:
                continue
            try:
                if job.frame.source_id != self.manager.workers[job.detection.camera_id].source_id:
                    continue
                crop = crop_identification(job.frame.image, job.detection.bbox, job.roi)
                with self._accelerator:
                    started = time.monotonic()
                    read = self.ocr.read(crop)
                    self.metrics.latency('ocr', time.monotonic() - started)
                validation = self.validator.validate(read)
                # Keep the failed event evidence in memory and retry storage. Capture
                # remains independent; a prolonged DB outage is visible in health.
                while not self._stop.is_set():
                    try:
                        with self._camera_locks[job.detection.camera_id]:
                            if job.frame.source_id != self.manager.workers[job.detection.camera_id].source_id:
                                break
                            additional_frames = {}
                            gate_cam_ids = [c.id for c in self.cameras if c.gate_id == job.gate_id]
                            for cam_id in gate_cam_ids:
                                if cam_id == job.detection.camera_id:
                                    continue
                                worker = self.manager.workers.get(cam_id)
                                if worker:
                                    latest = worker.latest()
                                    if latest and latest.image is not None:
                                        additional_frames[cam_id] = latest.image
                            result = self.events.observe(job, validation, additional_frames=additional_frames)
                        self.last_processing_error = None
                        if result:
                            with self._state_lock:
                                self._tracks[job.origin_key]['done'] = True
                            self.metrics.increment('events_created' if result[1] else 'evidence_associated')
                            self.metrics.latency('event', (utcnow() - job.started_at).total_seconds())
                        break
                    except Exception as exc:
                        self.last_processing_error = type(exc).__name__
                        self.metrics.increment('event_storage_errors')
                        log.error('event_storage_failed', extra={'camera_id': job.detection.camera_id, 'error_type': type(exc).__name__})
                        self._stop.wait(1)
            except Exception as exc:
                self.last_processing_error = type(exc).__name__
                self.metrics.increment('ocr_errors')
                log.error('ocr_failed', extra={'camera_id': job.detection.camera_id, 'error_type': type(exc).__name__})
            finally:
                with self._state_lock:
                    if job.origin_key in self._tracks:
                        self._tracks[job.origin_key]['pending'] = False
                    active = set(self._tracks)
                self.events.prune(active)
                self.queue.task_done()
