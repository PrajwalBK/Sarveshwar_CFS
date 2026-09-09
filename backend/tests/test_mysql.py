"""Opt-in integration against a disposable, user-provided MySQL test database."""
import os
from uuid import uuid4
import pytest
from sqlalchemy import delete, select
from sqlalchemy.engine import make_url
from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import mysql
from app.config.settings import CameraConfig
from app.database.database import Database
from app.database.models import Base, Camera, GateEvent, DetectionRecord, OCRResult, Snapshot, SystemLog
from app.database.repositories.gate import GateRepository
from tests.test_events_database import observation


def test_all_tables_compile_for_mysql():
    for table in Base.metadata.sorted_tables:
        assert 'CREATE TABLE' in str(CreateTable(table).compile(dialect=mysql.dialect()))


@pytest.mark.skipif(not os.getenv('TEST_MYSQL_URL'), reason='TEST_MYSQL_URL is not configured')
def test_mysql_transaction_and_dedup(snapshots, make_job):
    url = os.environ['TEST_MYSQL_URL']
    parsed = make_url(url)
    if parsed.drivername != 'mysql+pymysql' or not (parsed.database or '').startswith('gate_test_'):
        pytest.fail('Use a disposable MySQL database with a name starting gate_test_')
    db = Database(url)
    db.initialize()
    repo = GateRepository(db)
    camera_id = 'test-' + uuid4().hex[:16]
    gate_id = 'lane-' + uuid4().hex[:16]
    repo.sync_cameras([CameraConfig(id=camera_id, name='Test view', gate_id=gate_id, source_env='TEST_RTSP')])
    try:
        job = make_job(camera_id=camera_id, gate_id=gate_id, origin=uuid4().hex)
        first, created = repo.save_observation(observation(job), snapshots, 30)
        assert created
        assert repo.save_observation(observation(job), snapshots, 30) == (first, False)
        assert repo.event(first)['ocr_results'][0]['valid_check_digit']
        assert repo.events(camera_id=camera_id)['total'] == 1
    finally:
        # Only records generated for this random test lane are removed.
        with db.sessions.begin() as session:
            event_ids = list(session.scalars(select(GateEvent.id).where(GateEvent.gate_id == gate_id)))
            detection_ids = list(session.scalars(select(DetectionRecord.id).where(DetectionRecord.event_id.in_(event_ids))))
            session.execute(delete(OCRResult).where(OCRResult.detection_id.in_(detection_ids)))
            for model in (Snapshot, SystemLog, DetectionRecord):
                session.execute(delete(model).where(model.event_id.in_(event_ids)))
            session.execute(delete(GateEvent).where(GateEvent.id.in_(event_ids)))
            session.execute(delete(Camera).where(Camera.id == camera_id))
        db.close()
