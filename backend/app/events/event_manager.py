from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import logging
from app.domain import Observation

log = logging.getLogger('gate')


@dataclass
class Evidence:
    reads: list = field(default_factory=list)
    event_id: str | None = None
    last_seen: datetime | None = None
    last_sequence: int | None = None


class EventManager:
    """Called only by the OCR worker; commit succeeds before marking a track done."""
    def __init__(self, settings, repository, snapshots, dispatcher=None):
        self.settings, self.repository, self.snapshots = settings, repository, snapshots
        self.dispatcher = dispatcher
        self.evidence: dict[str, Evidence] = {}

    def prune(self, active_origins):
        self.evidence = {key: value for key, value in self.evidence.items() if key in active_origins}

    def observe(self, job, validation, additional_frames: dict | None = None):
        evidence = self.evidence.setdefault(job.origin_key, Evidence())
        if evidence.event_id:
            return evidence.event_id, False
        evidence.last_seen = job.detection.frame_timestamp
        # Retrying the same failed DB write is not another independent OCR vote.
        if evidence.last_sequence != job.frame.sequence:
            evidence.reads.append(validation)
            evidence.last_sequence = job.frame.sequence
        evidence.reads = evidence.reads[-self.settings.ocr_max_attempts:]
        good = [r for r in evidence.reads if r.valid_check_digit and r.confidence >= self.settings.ocr_min_confidence]
        votes = Counter(r.normalized_text for r in good)
        best, count = votes.most_common(1)[0] if votes else (None, 0)
        tied = sum(v == count for v in votes.values()) > 1
        confirmed = count >= self.settings.ocr_confirmations and not tied
        if not confirmed and len(evidence.reads) < self.settings.ocr_max_attempts:
            return None
        selected = max((r for r in good if r.normalized_text == best), key=lambda r: r.confidence) if confirmed and good else validation
        obs = Observation(job.origin_key, job.gate_id, job.track_id, job.detection, selected,
                          job.direction, job.frame.image, job.started_at, confirmed,
                          all_detections=list(getattr(job, 'all_detections', [])))

        event_id, created = self.repository.save_observation(
            obs, self.snapshots, self.settings.dedup_seconds, additional_frames
        )
        evidence.event_id = event_id
        if self.dispatcher and obs.confirmed:
            self.dispatcher.enqueue(obs, event_id, additional_frames=additional_frames)
        log.info('gate_event_created' if created else 'gate_evidence_associated',
                 extra={'camera_id': job.detection.camera_id, 'event_id': event_id})
        return event_id, created

