"""Unit tests for Prosper GateVision gate-events mapping."""

from app.prosper_gate_upload import (
    is_gatevision_test_recording_row,
    parse_gatevision_video_path,
    prosper_gate_uuid,
    record_to_gate_event_body,
    stage_to_gate_event_type,
)


def test_parse_gatevision_video_path():
    assert parse_gatevision_video_path("gatevision:gate-1:gate_arrival") == ("gate-1", "gate_arrival")
    assert parse_gatevision_video_path("gatevision:g-a:gate_pass") == ("g-a", "gate_pass")
    assert parse_gatevision_video_path("gatevision:test-myclip:gate_pass") == ("test-myclip", "gate_pass")
    assert parse_gatevision_video_path("yard/foo.mp4") == (None, None)


def test_is_gatevision_test_recording_row():
    assert is_gatevision_test_recording_row({"video_path": "gatevision:test-foo:gate_pass"})
    assert not is_gatevision_test_recording_row({"video_path": "gatevision:test:old.mp4"})
    assert not is_gatevision_test_recording_row({"video_path": "gatevision:gate-1:gate_arrival"})


def test_stage_to_gate_event_type():
    assert stage_to_gate_event_type("gate_arrival") == 1
    assert stage_to_gate_event_type("gate_departure") == 2
    assert stage_to_gate_event_type("gate_pass") == 1
    assert stage_to_gate_event_type("candidate") is None


def test_prosper_gate_uuid_map_overrides_uuid5():
    fixed = "33333333-3333-3333-3333-333333333333"
    m = {"gate-1": fixed}
    assert prosper_gate_uuid("gate-1", m) == fixed.lower()
    u = prosper_gate_uuid("gate-999", None)
    assert len(u) == 36
    v = prosper_gate_uuid("gate-999", None)
    assert u == v


def test_record_to_gate_event_body_arrival():
    body = record_to_gate_event_body(
        {
            "id": 7,
            "video_path": "gatevision:gate-1:gate_arrival",
            "licence_plate_trailer": "ABCD 12345",
            "confidence": 0.9,
            "timestamp": "2026-05-04T12:00:00Z",
        },
        device_id_raw="edge-test",
        gate_id_map=None,
    )
    assert body is not None
    assert body["eventType"] == 1
    assert body["sourceSystem"] == "GateVision"
    assert body["trailerNumber"] == "12345"
    assert body["scac"] == "ABCD"
    assert body["damageFlag"] is False


def test_record_to_gate_event_body_skips_candidate():
    assert (
        record_to_gate_event_body(
            {"id": 1, "video_path": "gatevision:gate-1:candidate", "licence_plate_trailer": "X"},
            device_id_raw="x",
        )
        is None
    )
