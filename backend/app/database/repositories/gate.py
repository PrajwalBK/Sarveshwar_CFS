from dataclasses import asdict
from datetime import datetime, timedelta
from threading import Lock
from uuid import uuid4
from sqlalchemy import func, select

from app.database.models import Camera, DetectionRecord, GateEvent, OCRResult, Snapshot, SystemLog
from app.domain import Observation


def serialize(record):
    result = {column.name: getattr(record, column.name) for column in record.__table__.columns}
    return {key: value.isoformat() + 'Z' if isinstance(value, datetime) else value for key, value in result.items()}


class GateRepository:
    def __init__(self, db):
        self.db = db
        self._write_lock = Lock()

    def sync_cameras(self, cameras):
        with self.db.sessions.begin() as session:
            for camera in cameras:
                current = session.get(Camera, camera.id)
                if current is None:
                    current = Camera(id=camera.id)
                    session.add(current)
                current.name = camera.name
                current.gate_id = camera.gate_id
                current.source_env = camera.source_env
                current.enabled = camera.enabled
                current.configuration = camera.model_dump(mode='json')

    def save_observation(self, obs: Observation, snapshots, dedup_seconds: float, additional_frames: dict | None = None):
        """One writer per service; DB uniqueness also protects retried track writes."""
        created_paths = []
        try:
            with self._write_lock, self.db.sessions.begin() as session:
                prior = session.scalar(select(DetectionRecord).where(DetectionRecord.origin_key == obs.origin_key))
                if prior:
                    return prior.event_id, False
                number = obs.ocr.normalized_text if obs.confirmed else None
                delta = timedelta(seconds=dedup_seconds)
                existing = None
                if number:
                    existing = session.scalar(select(GateEvent).where(
                        GateEvent.gate_id == obs.gate_id,
                        GateEvent.container_number == number,
                        GateEvent.event_type == obs.direction,
                        GateEvent.timestamp >= obs.detection.frame_timestamp - delta,
                        GateEvent.timestamp <= obs.detection.frame_timestamp + delta,
                    ).order_by(GateEvent.timestamp.desc()))
                    if not existing:
                        unconfirmed = session.scalar(select(GateEvent).where(
                            GateEvent.gate_id == obs.gate_id,
                            GateEvent.container_number.is_(None),
                            GateEvent.event_type == obs.direction,
                            GateEvent.timestamp >= obs.detection.frame_timestamp - delta,
                            GateEvent.timestamp <= obs.detection.frame_timestamp + delta,
                        ).order_by(GateEvent.timestamp.desc()))
                        if unconfirmed:
                            unconfirmed.container_number = number
                            unconfirmed.status = 'CONFIRMED'
                            existing = unconfirmed
                else:
                    existing = session.scalar(select(GateEvent).where(
                        GateEvent.gate_id == obs.gate_id,
                        GateEvent.event_type == obs.direction,
                        GateEvent.timestamp >= obs.detection.frame_timestamp - delta,
                        GateEvent.timestamp <= obs.detection.frame_timestamp + delta,
                    ).order_by(GateEvent.timestamp.desc()))
                created = existing is None
                event_id = existing.id if existing else str(uuid4())
                if created:
                    session.add(GateEvent(
                        id=event_id, gate_id=obs.gate_id, timestamp=obs.detection.frame_timestamp,
                        container_number=number, event_type=obs.direction,
                        status='CONFIRMED' if obs.confirmed else 'NEEDS_REVIEW',
                    ))
                    session.flush()
                detection_id = str(uuid4())
                session.add(DetectionRecord(
                    id=detection_id, event_id=event_id, camera_id=obs.detection.camera_id,
                    origin_key=obs.origin_key, track_id=obs.track_id,
                    class_name=obs.detection.class_name, confidence=obs.detection.confidence,
                    bbox=list(obs.detection.bbox), timestamp=obs.detection.frame_timestamp,
                ))
                session.flush()
                session.add(OCRResult(id=str(uuid4()), detection_id=detection_id, **asdict(obs.ocr)))

                # Also save any concurrent detections on this frame (such as Feet!)
                if getattr(obs, 'all_detections', None):
                    for extra in obs.all_detections:
                        if extra.class_name != obs.detection.class_name:
                            extra_origin = f'{obs.origin_key}:{extra.class_name}:{int(extra.bbox[0])}'
                            existing_extra = session.scalar(select(DetectionRecord).where(
                                DetectionRecord.origin_key == extra_origin
                            ))
                            if not existing_extra:
                                session.add(DetectionRecord(
                                    id=str(uuid4()), event_id=event_id, camera_id=extra.camera_id,
                                    origin_key=extra_origin, track_id=obs.track_id,
                                    class_name=extra.class_name, confidence=extra.confidence,
                                    bbox=list(extra.bbox), timestamp=extra.frame_timestamp,
                                ))


                # 1. Primary snapshot for this observation
                existing_cam_snaps = session.scalars(select(Snapshot).where(
                    Snapshot.event_id == event_id,
                    Snapshot.camera_id == obs.detection.camera_id
                )).all()
                prior_detection = session.scalar(select(DetectionRecord).where(
                    DetectionRecord.event_id == event_id,
                    DetectionRecord.camera_id == obs.detection.camera_id,
                    DetectionRecord.id != detection_id
                ))
                same_track_recorded = session.scalar(select(DetectionRecord).where(
                    DetectionRecord.event_id == event_id,
                    DetectionRecord.camera_id == obs.detection.camera_id,
                    DetectionRecord.track_id == obs.track_id,
                    DetectionRecord.id != detection_id
                ))

                if existing_cam_snaps and not prior_detection:
                    # Camera was previously stored as background frame from additional_frames;
                    # upgrade it now with the actual detection bounding box!
                    snap = existing_cam_snaps[0]
                    new_path = snapshots.save(snap.id, obs, event_id)
                    snap.image_path = new_path
                    snap.timestamp = obs.detection.frame_timestamp
                elif not existing_cam_snaps or not same_track_recorded:
                    # First time this camera/track is recorded for this event
                    snapshot_id = str(uuid4())
                    image_path = snapshots.save(snapshot_id, obs, event_id)
                    created_paths.append(image_path)
                    session.add(Snapshot(id=snapshot_id, event_id=event_id, camera_id=obs.detection.camera_id,
                                         image_path=image_path, timestamp=obs.detection.frame_timestamp))
                elif obs.confirmed and existing_cam_snaps:
                    snap = existing_cam_snaps[-1]
                    new_path = snapshots.save(snap.id, obs, event_id)
                    snap.image_path = new_path
                    snap.timestamp = obs.detection.frame_timestamp

                # 2. Multi-camera snapshots for all other cameras covering this gate event
                if additional_frames:
                    for other_cam_id, other_img in additional_frames.items():
                        if other_cam_id == obs.detection.camera_id or other_img is None:
                            continue
                        already_saved = session.scalar(select(Snapshot).where(
                            Snapshot.event_id == event_id,
                            Snapshot.camera_id == other_cam_id
                        ))
                        if not already_saved:
                            other_snap_id = str(uuid4())
                            other_path = snapshots.save_camera_frame(
                                other_snap_id, event_id, other_cam_id, other_img,
                                direction=obs.direction,
                                text=obs.ocr.normalized_text if obs.confirmed else ''
                            )
                            created_paths.append(other_path)
                            session.add(Snapshot(
                                id=other_snap_id, event_id=event_id, camera_id=other_cam_id,
                                image_path=other_path, timestamp=obs.detection.frame_timestamp
                            ))

                session.add(SystemLog(id=str(uuid4()), event_id=event_id, level='INFO',
                                      event_name='gate_event_created' if created else 'gate_evidence_associated',
                                      context={'camera_id': obs.detection.camera_id, 'gate_id': obs.gate_id}))

            try:
                full_event = self.event(event_id)
                if full_event and hasattr(snapshots, 'save_event_json'):
                    snapshots.save_event_json(event_id, full_event)
            except Exception:
                pass

            return event_id, created
        except Exception:
            for p in created_paths:
                try:
                    snapshots.remove(p)
                except Exception:
                    pass
            raise

    def events(self, limit=50, offset=0, container=None, camera_id=None, event_type=None, status=None):
        query = select(GateEvent)
        if container:
            query = query.where(GateEvent.container_number == container.upper())
        if camera_id:
            query = query.where(GateEvent.id.in_(select(DetectionRecord.event_id).where(DetectionRecord.camera_id == camera_id)))
        if event_type:
            query = query.where(GateEvent.event_type == event_type)
        if status:
            query = query.where(GateEvent.status == status)
        with self.db.sessions() as session:
            total = session.scalar(select(func.count()).select_from(query.subquery()))
            items = [serialize(e) for e in session.scalars(query.order_by(GateEvent.timestamp.desc(), GateEvent.id).limit(limit).offset(offset))]
            from app.ocr.validator import parse_feet_size
            import re
            for item in items:
                dets = list(session.scalars(select(DetectionRecord).where(DetectionRecord.event_id == item['id'])))
                has_feet = any(d.class_name.lower() == 'feet' for d in dets)
                ocrs = list(session.scalars(select(OCRResult).where(OCRResult.detection_id.in_([d.id for d in dets])))) if dets else []
                parsed_size = None
                size_code = None
                for o in ocrs:
                    text_to_check = f"{o.normalized_text} {o.raw_text}"
                    parsed = parse_feet_size(text_to_check)
                    if parsed:
                        parsed_size = parsed
                        m = re.search(r'(?<![A-Z0-9])([1-4LMN][0-9][GVRHBUTP][0-9A-Z])(?![A-Z0-9])', text_to_check.upper())
                        if m:
                            size_code = m.group(1)
                        break
                item['container_size'] = parsed_size or ('FEET' if has_feet else None)
                item['size_code'] = size_code
            return {'items': items, 'total': total, 'limit': limit, 'offset': offset}

    def event(self, event_id):
        with self.db.sessions() as session:
            row = session.get(GateEvent, event_id)
            if row is None:
                return None
            result = serialize(row)
            detections = list(session.scalars(select(DetectionRecord).where(DetectionRecord.event_id == event_id)))
            result['detections'] = [serialize(d) for d in detections]
            result['ocr_results'] = [serialize(o) for o in session.scalars(select(OCRResult).where(
                OCRResult.detection_id.in_([d.id for d in detections]))) ]
            result['snapshots'] = [serialize(s) for s in session.scalars(select(Snapshot).where(Snapshot.event_id == event_id))]
            
            from app.ocr.validator import parse_feet_size
            import re
            container_size = None
            size_code = None
            for ocr in result['ocr_results']:
                text_to_check = f"{ocr.get('normalized_text', '')} {ocr.get('raw_text', '')}"
                parsed = parse_feet_size(text_to_check)
                if parsed:
                    container_size = parsed
                    m = re.search(r'(?<![A-Z0-9])([1-4LMN][0-9][GVRHBUTP][0-9A-Z])(?![A-Z0-9])', text_to_check.upper())
                    if m:
                        size_code = m.group(1)
                    break
            if not container_size and any(d.get('class_name', '').lower() == 'feet' for d in result['detections']):
                container_size = 'FEET DETECTED'
            result['container_size'] = container_size
            result['size_code'] = size_code
            return result

    def ocr_results(self, limit=50, offset=0):
        with self.db.sessions() as session:
            return [serialize(o) for o in session.scalars(select(OCRResult).order_by(OCRResult.timestamp.desc()).limit(limit).offset(offset))]

    def snapshot(self, snapshot_id):
        with self.db.sessions() as session:
            row = session.get(Snapshot, snapshot_id)
            return serialize(row) if row else None
