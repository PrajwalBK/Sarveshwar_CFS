from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uuid


@dataclass
class CameraObservation:
    camera_id: str
    camera_role: str
    gate_id: str
    ts: datetime
    track_id: int
    text: str
    conf: float
    bbox: List[float]
    evidence_image_path: Optional[str] = None


@dataclass
class GatePassCandidate:
    gate_pass_id: str
    gate_id: str
    started_at: datetime
    updated_at: datetime
    observations: List[CameraObservation] = field(default_factory=list)


class GateFusionEngine:
    """
    Correlates per-camera observations into a single GateVision pass event.
    """

    def __init__(
        self,
        fusion_window_seconds: float = 3.0,
        finalize_after_seconds: float = 1.5,
        min_roles_required: int = 1,
        confirm_conf_threshold: float = 0.65,
    ):
        self.fusion_window = timedelta(seconds=float(fusion_window_seconds))
        self.finalize_after = timedelta(seconds=float(finalize_after_seconds))
        self.min_roles_required = max(1, int(min_roles_required))
        self.confirm_conf_threshold = float(confirm_conf_threshold)
        self._candidates: Dict[str, GatePassCandidate] = {}

    def add_observation(self, obs: CameraObservation) -> None:
        candidate = self._find_candidate(obs)
        if candidate is None:
            candidate = GatePassCandidate(
                gate_pass_id=str(uuid.uuid4()),
                gate_id=obs.gate_id,
                started_at=obs.ts,
                updated_at=obs.ts,
                observations=[],
            )
            self._candidates[candidate.gate_pass_id] = candidate
        candidate.observations.append(obs)
        candidate.updated_at = obs.ts

    def collect_ready_events(self, now: Optional[datetime] = None) -> List[dict]:
        now = now or datetime.utcnow()
        ready: List[dict] = []
        to_remove: List[str] = []

        for gate_pass_id, candidate in self._candidates.items():
            if now - candidate.updated_at < self.finalize_after:
                continue
            to_remove.append(gate_pass_id)
            event = self._finalize_candidate(candidate)
            if event is not None:
                ready.append(event)

        for gate_pass_id in to_remove:
            self._candidates.pop(gate_pass_id, None)

        return ready

    def get_active_candidates(self) -> List[dict]:
        items = []
        for candidate in self._candidates.values():
            roles = sorted({o.camera_role for o in candidate.observations})
            items.append(
                {
                    "gate_pass_id": candidate.gate_pass_id,
                    "gate_id": candidate.gate_id,
                    "event_type": "arrival_candidate",
                    "started_at": candidate.started_at.isoformat(),
                    "updated_at": candidate.updated_at.isoformat(),
                    "observations_count": len(candidate.observations),
                    "camera_roles_seen": roles,
                    "ready_for_finalize": len(roles) >= self.min_roles_required,
                }
            )
        return items

    def _find_candidate(self, obs: CameraObservation) -> Optional[GatePassCandidate]:
        nearest: Optional[GatePassCandidate] = None
        nearest_dt: Optional[timedelta] = None
        for candidate in self._candidates.values():
            if candidate.gate_id != obs.gate_id:
                continue
            dt = abs(candidate.updated_at - obs.ts)
            if dt > self.fusion_window:
                continue
            if nearest is None or dt < nearest_dt:
                nearest = candidate
                nearest_dt = dt
        return nearest

    def _finalize_candidate(self, candidate: GatePassCandidate) -> Optional[dict]:
        roles = {o.camera_role for o in candidate.observations}
        if len(roles) < self.min_roles_required:
            return None

        text_scores: Dict[str, float] = {}
        for obs in candidate.observations:
            text = (obs.text or "").strip().upper()
            if not text:
                continue
            weight = obs.conf if obs.conf is not None else 0.0
            if obs.camera_role in ("gate_rear", "gate_front"):
                weight *= 1.2
            text_scores[text] = text_scores.get(text, 0.0) + weight

        best_text = ""
        best_score = 0.0
        if text_scores:
            best_text, best_score = max(text_scores.items(), key=lambda kv: kv[1])

        direction_votes = {"INBOUND": 0, "OUTBOUND": 0}
        for obs in candidate.observations:
            role = (obs.camera_role or "").lower()
            gid = (obs.gate_id or "").lower()
            cid = (obs.camera_id or "").lower()

            if any(k in role or k in gid or k in cid for k in ("in", "entry", "front")):
                direction_votes["INBOUND"] += 1
            elif any(k in role or k in gid or k in cid for k in ("out", "exit", "rear")):
                direction_votes["OUTBOUND"] += 1

        if direction_votes["INBOUND"] > direction_votes["OUTBOUND"]:
            direction = "INBOUND"
        elif direction_votes["OUTBOUND"] > direction_votes["INBOUND"]:
            direction = "OUTBOUND"
        elif "gate-in" in candidate.gate_id.lower():
            direction = "INBOUND"
        elif "gate-out" in candidate.gate_id.lower():
            direction = "OUTBOUND"
        else:
            direction = "INBOUND"

        confidence = min(1.0, best_score / max(1, len(candidate.observations)))
        status = "confirmed" if confidence >= self.confirm_conf_threshold and best_text else "needs_review"

        evidence = []
        for obs in candidate.observations:
            evidence.append(
                {
                    "camera_id": obs.camera_id,
                    "camera_role": obs.camera_role,
                    "track_id": obs.track_id,
                    "text": obs.text,
                    "conf": obs.conf,
                    "bbox": obs.bbox,
                    "ts_iso": obs.ts.isoformat(),
                    "evidence_image_path": obs.evidence_image_path,
                }
            )

        return {
            "event_type": "gate_pass",
            "gate_pass_id": candidate.gate_pass_id,
            "gate_id": candidate.gate_id,
            "direction": direction,
            "ts_iso": candidate.updated_at.isoformat(),
            "event_time_start": candidate.started_at.isoformat(),
            "event_time_end": candidate.updated_at.isoformat(),
            "trailer_id": best_text or None,
            "conf": confidence,
            "status": status,
            "camera_roles_seen": sorted(list(roles)),
            "evidence": evidence,
        }

