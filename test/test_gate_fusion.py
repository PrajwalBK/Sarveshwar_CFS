from datetime import datetime, timedelta

from app.gatevision.fusion import CameraObservation, GateFusionEngine


def test_gate_fusion_emits_event_for_multi_camera_pass():
    engine = GateFusionEngine(
        fusion_window_seconds=3.0,
        finalize_after_seconds=0.2,
        min_roles_required=2,
    )
    now = datetime.utcnow()

    engine.add_observation(
        CameraObservation(
            camera_id="gate-front-01",
            camera_role="gate_front",
            gate_id="gate-1",
            ts=now,
            track_id=101,
            text="TRL1015",
            conf=0.9,
            bbox=[0, 0, 10, 10],
        )
    )
    engine.add_observation(
        CameraObservation(
            camera_id="gate-left-01",
            camera_role="gate_left",
            gate_id="gate-1",
            ts=now + timedelta(milliseconds=120),
            track_id=55,
            text="TRL1015",
            conf=0.7,
            bbox=[1, 1, 11, 11],
        )
    )

    events = engine.collect_ready_events(now=now + timedelta(seconds=1))
    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "gate_pass"
    assert event["gate_id"] == "gate-1"
    assert event["trailer_id"] == "TRL1015"
    assert event["status"] in ("confirmed", "needs_review")

