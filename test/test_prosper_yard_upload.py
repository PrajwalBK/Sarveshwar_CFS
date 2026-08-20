"""Unit tests for Prosper YardVision upload mapping."""

from app.prosper_yard_upload import (
    is_yardvision_sqlite_record,
    prosper_device_uuid,
    prosper_event_uuid,
    record_to_trailer_location_body,
)


def test_prosper_device_uuid_stable_for_non_uuid():
    a = prosper_device_uuid("edge-box-7")
    b = prosper_device_uuid("edge-box-7")
    assert a == b
    assert len(a) == 36


def test_prosper_event_uuid_stable():
    e1 = prosper_event_uuid(42, "dev", "2026-01-01T00:00:00")
    e2 = prosper_event_uuid(42, "dev", "2026-01-01T00:00:00")
    assert e1 == e2


def test_is_yardvision_sqlite_record_excludes_gate():
    assert is_yardvision_sqlite_record({"video_path": "foo.mp4"})
    assert not is_yardvision_sqlite_record({"video_path": "gatevision:gate-1:candidate"})


def test_record_to_body_skips_unknown_plate():
    assert (
        record_to_trailer_location_body(
            {"id": 1, "licence_plate_trailer": "UNKNOWN", "confidence": 0.5},
            device_id_raw="x",
        )
        is None
    )


def test_record_to_body_minimal():
    body = record_to_trailer_location_body(
        {
            "id": 99,
            "licence_plate_trailer": "TRL9",
            "latitude": 33.1,
            "longitude": -112.0,
            "confidence": 0.88,
            "timestamp": "2026-05-04T12:00:00Z",
        },
        device_id_raw="my-device",
    )
    assert body is not None
    assert body["trailerNumber"] == "TRL9"
    assert body["sourceSystem"] == "YardVision"
    assert body["latitude"] == 33.1
    assert body["longitude"] == -112.0


def test_extract_trailer_and_scac_spacing():
    from app.prosper_yard_upload import _extract_trailer_and_scac
    
    # 1. Contiguous text without spaces
    t1, s1 = _extract_trailer_and_scac("JBHU322099")
    assert t1 == "322099"
    assert s1 == "JBHU"

    # 2. Contiguous text with separator character
    t2, s2 = _extract_trailer_and_scac("JBHU - 322099")
    assert t2 == "322099"
    assert s2 == "JBHU"

    # 3. Space-separated digits (should not be joined)
    t3, s3 = _extract_trailer_and_scac("Container HUNT Intermodal 1889 68565")
    assert t3 == "68565"
    assert s3 == "JBHU"

    # 4. Long prefix that is not a SCAC (should return trailer number and None for SCAC)
    t4, s4 = _extract_trailer_and_scac("CANINE TRANSFER LAWRENCE WEAVER R53275")
    assert t4 == "53275"
    assert s4 is None


