"""Deterministic hardware-free API smoke fixture; never uses production MySQL/cameras.

python -m tools.simulate --serve
Use real recorded-video settings to measure model accuracy; this fixture injects AI.
"""
import argparse
from datetime import timedelta
from pathlib import Path
import tempfile
import numpy as np

from app.config.settings import CameraConfig, Settings
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.domain import Detection, Frame, OCRRead, utcnow
from app.events.event_manager import EventManager
from app.main import create_app
from app.ocr.validator import ContainerValidator
from app.runtime import OCRJob
from app.snapshots.snapshot_manager import SnapshotManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    # A fresh directory prevents deletion/overwriting of any existing DB or images.
    root = Path(tempfile.mkdtemp(prefix='gate-simulation-'))
    settings = Settings(_env_file=None, deployment_mode='test', database_url='sqlite+pysqlite:///' + str(root / 'simulation.db'),
                        snapshot_directory=root / 'snapshots', upload_directory=root / 'uploads', pipeline_enabled=False)
    cameras = [CameraConfig(id=f'simulation-{i}', name=f'Simulation view {i}', gate_id=f'simulation-lane-{(i-1)//2}',
                            source_env=f'SIMULATION_DISABLED_{i}', enabled=False) for i in range(1, 5)]
    db = Database(settings.database_url.get_secret_value())
    db.initialize()
    repository = GateRepository(db)
    repository.sync_cameras(cameras)
    manager = EventManager(settings, repository, SnapshotManager(settings.snapshot_directory))
    validation = ContainerValidator().validate(OCRRead('CSQU3054383', .94))
    import cv2
    ts = utcnow()
    for i, camera in enumerate(cameras):
        image = np.full((360, 640, 3), (39, 49, 59), dtype=np.uint8)
        cv2.rectangle(image, (80, 70), (560, 285), (128, 98, 62), -1)
        cv2.putText(image, 'SYNTHETIC TEST FIXTURE', (100, 120), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        cv2.putText(image, 'CSQU 305438 3', (140, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        for sequence in (1, 2):
            timestamp = ts + timedelta(milliseconds=sequence)
            detection = Detection(camera.id, timestamp, 'container', .92, (80, 70, 560, 285))
            job = OCRJob(f'simulation:{camera.id}:1', camera.gate_id, 1, detection,
                         Frame(camera.id, timestamp, sequence, image), 'UNKNOWN', ts, (0, 0, 1, 1))
            manager.observe(job, validation)
    print(f'Synthetic fixture: {repository.events()["total"]} gate events from four views. Files: {root}')
    if args.serve:
        import uvicorn
        uvicorn.run(create_app(settings, db, cameras), host='127.0.0.1', port=args.port, workers=1, access_log=False)
    else:
        db.close()


if __name__ == '__main__':
    main()
