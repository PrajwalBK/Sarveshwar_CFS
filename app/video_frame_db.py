"""
Database Manager for Video Frame Records (split YardVision / GateVision).

Two tables in data/video_frames.db:
  - yardvision_records: trailer locations from yard cameras (parking-spot resolution flow).
                        Rows are inserted is_processed=0; data_processor flips to 1 after
                        spot assignment; upload thread reads is_processed=1 and deletes.

  - gatevision_records: gate events from gate cameras (live + test-mode).
                        source='live' | 'test'. Uploaded as Prosper gate-events and
                        deleted after acknowledgement. No is_processed flow — gate rows
                        skip the data processor entirely (they don't park in spots).

Routing: insert_frame_record(...) keeps a single back-compat entry-point that inspects
video_path and dispatches. New callers should use insert_yardvision_record() or
insert_gatevision_event() directly for clarity.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import threading
from app.app_logger import get_logger

logger = get_logger(__name__)


def _parse_legacy_gate_video_path(video_path: str) -> Dict[str, Optional[str]]:
    """Parse legacy `video_path` strings into structured gate columns.

    Live shape:  ``gatevision:<gate_id>:<event_type>``  (e.g. gatevision:gate-1:gate_pass)
    Test shape:  ``gatevision:test-<video_stem>:<event_type>``  (event_type is usually gate_pass)

    Returns a dict with: gate_id, event_type, source, test_video_stem.
    """
    out: Dict[str, Optional[str]] = {
        "gate_id": None,
        "event_type": None,
        "source": "live",
        "test_video_stem": None,
    }
    if not video_path or not video_path.startswith("gatevision:"):
        return out
    # Split into at most 3 parts: 'gatevision', '<middle>', '<event_type>'
    parts = video_path.split(":", 2)
    if len(parts) < 3:
        return out
    middle, event_type = parts[1], parts[2]
    out["event_type"] = event_type or None
    if middle.startswith("test-"):
        out["source"] = "test"
        out["test_video_stem"] = middle[len("test-"):] or None
    else:
        out["source"] = "live"
        out["gate_id"] = middle or None
    return out


class VideoFrameDB:
    """Manages SQLite storage for YardVision and GateVision results in two tables."""

    def __init__(self, db_path: str = "data/video_frames.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._initialize_database()

    # ------------------------------------------------------------------ schema

    def _initialize_database(self):
        """Create per-pipeline tables; drop the legacy combined table if present.

        The old `video_frame_records` table is dropped on first run (per design decision —
        existing dev data is disposable). Both new tables use IF NOT EXISTS, so this is
        idempotent across restarts.
        """
        create_yard_sql = """
        CREATE TABLE IF NOT EXISTS yardvision_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            licence_plate_trailer TEXT,
            latitude REAL,
            longitude REAL,
            speed REAL,
            barrier REAL,
            confidence REAL,
            image_path TEXT,
            camera_id TEXT,
            video_path TEXT,
            frame_number INTEGER,
            track_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_processed BOOLEAN DEFAULT 0,
            created_on DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_on DATETIME DEFAULT CURRENT_TIMESTAMP,
            assigned_spot_id TEXT,
            assigned_spot_name TEXT,
            assigned_distance_ft REAL,
            processed_comment TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_yardvision_unprocessed
            ON yardvision_records(is_processed, created_on)
            WHERE is_processed = 0;
        CREATE INDEX IF NOT EXISTS idx_yardvision_camera
            ON yardvision_records(camera_id, created_on);
        CREATE INDEX IF NOT EXISTS idx_yardvision_plate
            ON yardvision_records(licence_plate_trailer, is_processed);
        """

        create_gate_sql = """
        CREATE TABLE IF NOT EXISTS gatevision_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            licence_plate_trailer TEXT,
            latitude REAL,
            longitude REAL,
            speed REAL,
            confidence REAL,
            image_path TEXT,
            camera_id TEXT,
            frame_number INTEGER,
            track_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_on DATETIME DEFAULT CURRENT_TIMESTAMP,
            gate_id TEXT,
            event_type TEXT,
            source TEXT,
            test_video_stem TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_gatevision_created
            ON gatevision_records(created_on);
        CREATE INDEX IF NOT EXISTS idx_gatevision_event_type
            ON gatevision_records(event_type, created_on);
        CREATE INDEX IF NOT EXISTS idx_gatevision_source
            ON gatevision_records(source, created_on);
        """

        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.executescript(create_yard_sql)
                conn.executescript(create_gate_sql)
                # Drop legacy combined table if it survives from before the split.
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='video_frame_records'"
                )
                if cursor.fetchone():
                    cursor.execute("DROP TABLE video_frame_records")
                    print("[VideoFrameDB] Dropped legacy table video_frame_records (split into yardvision_records / gatevision_records).")
                conn.commit()
                print(f"[VideoFrameDB] Database initialized: {self.db_path}")
            except Exception as e:
                print(f"[VideoFrameDB] Error initializing database: {e}")
                conn.rollback()
                raise
            finally:
                conn.close()

    # ----------------------------------------------------------------- inserts

    def insert_frame_record(
        self,
        licence_plate_trailer: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        speed: Optional[float] = None,
        barrier: Optional[float] = None,
        confidence: float = 0.0,
        image_path: str = "",
        camera_id: str = "",
        video_path: str = "",
        frame_number: int = 0,
        track_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """Back-compat router: dispatches to yard or gate table based on video_path.

        Existing call sites pass `video_path` strings shaped either as a filesystem path
        (yard) or `gatevision:...` (gate). We parse and route here so callers don't change.
        New code should prefer insert_yardvision_record() / insert_gatevision_event().
        """
        vp_low = (video_path or "").lower()
        cid_low = (camera_id or "").lower()
        is_gate = vp_low.startswith("gatevision:") or "gatevision" in vp_low or cid_low.startswith("gate") or "gate_" in cid_low
        if is_gate:
            parsed = _parse_legacy_gate_video_path(video_path)
            gate_id = parsed["gate_id"] or (camera_id if (cid_low.startswith("gate") or "gate_" in cid_low) else "gate-1")
            event_type = parsed["event_type"] or "gate_pass"
            return self.insert_gatevision_event(
                licence_plate_trailer=licence_plate_trailer,
                latitude=latitude,
                longitude=longitude,
                speed=speed,
                confidence=confidence,
                image_path=image_path,
                camera_id=camera_id,
                frame_number=frame_number,
                track_id=track_id,
                timestamp=timestamp,
                gate_id=gate_id,
                event_type=event_type,
                source=parsed["source"] or "live",
                test_video_stem=parsed["test_video_stem"],
            )
        return self.insert_yardvision_record(
            licence_plate_trailer=licence_plate_trailer,
            latitude=latitude,
            longitude=longitude,
            speed=speed,
            barrier=barrier,
            confidence=confidence,
            image_path=image_path,
            camera_id=camera_id,
            video_path=video_path,
            frame_number=frame_number,
            track_id=track_id,
            timestamp=timestamp,
        )

    def insert_yardvision_record(
        self,
        licence_plate_trailer: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        speed: Optional[float] = None,
        barrier: Optional[float] = None,
        confidence: float = 0.0,
        image_path: str = "",
        camera_id: str = "",
        video_path: str = "",
        frame_number: int = 0,
        track_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """Insert a YardVision record (trailer location pending spot assignment)."""
        if licence_plate_trailer:
            try:
                from app.container_utils import is_valid_trailer_id
                if not is_valid_trailer_id(licence_plate_trailer):
                    logger.info(f"[VideoFrameDB] Rejecting invalid/hallucinated trailer record: {licence_plate_trailer!r}")
                    return 0
            except ImportError:
                pass

        if timestamp is None:
            timestamp = datetime.utcnow()
        sql = """
        INSERT INTO yardvision_records (
            licence_plate_trailer, latitude, longitude, speed, barrier,
            confidence, image_path, camera_id, video_path, frame_number,
            track_id, timestamp, is_processed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(sql, (
                    licence_plate_trailer,
                    latitude,
                    longitude,
                    speed,
                    barrier,
                    confidence,
                    image_path,
                    camera_id,
                    video_path,
                    frame_number,
                    track_id,
                    timestamp.isoformat(),
                ))
                rid = cursor.lastrowid
                conn.commit()
                return rid
            except Exception as e:
                print(f"[VideoFrameDB] Error inserting yardvision record: {e}")
                conn.rollback()
                raise
            finally:
                conn.close()

    def insert_gatevision_event(
        self,
        licence_plate_trailer: str,
        *,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        speed: Optional[float] = None,
        confidence: float = 0.0,
        image_path: str = "",
        camera_id: str = "",
        frame_number: int = 0,
        track_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        gate_id: Optional[str] = None,
        event_type: Optional[str] = None,
        source: str = "live",
        test_video_stem: Optional[str] = None,
    ) -> int:
        """Insert a GateVision event row.

        source='live' rows: gate_id required, test_video_stem NULL.
        source='test' rows: test_video_stem required, gate_id may be NULL.
        event_type: 'gate_arrival' | 'gate_departure' | 'gate_pass' | 'candidate'
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        sql = """
        INSERT INTO gatevision_records (
            licence_plate_trailer, latitude, longitude, speed, confidence,
            image_path, camera_id, frame_number, track_id, timestamp,
            gate_id, event_type, source, test_video_stem
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(sql, (
                    licence_plate_trailer,
                    latitude,
                    longitude,
                    speed,
                    confidence,
                    image_path,
                    camera_id,
                    frame_number,
                    track_id,
                    timestamp.isoformat(),
                    gate_id,
                    event_type,
                    source,
                    test_video_stem,
                ))
                rid = cursor.lastrowid
                conn.commit()
                return rid
            except Exception as e:
                print(f"[VideoFrameDB] Error inserting gatevision event: {e}")
                conn.rollback()
                raise
            finally:
                conn.close()

    # ------------------------------------------------------------ yard reads

    def get_unprocessed_records(
        self,
        limit: int = 50,
        cutoff_seconds: int = 10,
        camera_id: Optional[str] = None,
    ) -> List[Dict]:
        """Yardvision rows awaiting spot assignment (data processor input)."""
        if limit <= 0:
            return []
        query = """
        SELECT id, licence_plate_trailer, latitude, longitude, speed, barrier,
               confidence, image_path, camera_id, video_path, frame_number,
               track_id, timestamp, created_on
        FROM yardvision_records
        WHERE is_processed = 0
          AND datetime(created_on) < datetime('now', '-' || ? || ' seconds')
        """
        params: List = [cutoff_seconds]
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        query += " ORDER BY created_on DESC LIMIT ?"
        params.append(limit)

        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
            except Exception as e:
                print(f"[VideoFrameDB] Error fetching unprocessed records: {e}")
                return []
            finally:
                conn.close()

    def mark_as_processed(
        self,
        record_id: int,
        comment: str = "",
        assigned_spot_id: Optional[str] = None,
        assigned_spot_name: Optional[str] = None,
        assigned_distance_ft: Optional[float] = None,
    ):
        """Flip a yardvision row to is_processed=1 after spot resolution."""
        sql = """
        UPDATE yardvision_records
        SET is_processed = 1,
            updated_on = CURRENT_TIMESTAMP,
            assigned_spot_id = ?,
            assigned_spot_name = ?,
            assigned_distance_ft = ?,
            processed_comment = ?
        WHERE id = ?
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(sql, (
                    assigned_spot_id,
                    assigned_spot_name,
                    assigned_distance_ft,
                    comment,
                    record_id,
                ))
                conn.commit()
            except Exception as e:
                print(f"[VideoFrameDB] Error marking record as processed: {e}")
                conn.rollback()
            finally:
                conn.close()

    def get_all_records(
        self,
        limit: int = 50,
        offset: int = 0,
        is_processed: Optional[bool] = None,
        camera_id: Optional[str] = None,
    ) -> List[Dict]:
        """Yardvision-only read (used by upload thread + dashboard endpoints).

        The dashboard reads (events, inventory, yard view, reports, KPIs) all assume
        yard semantics — assigned_spot_*, parking lanes, etc. — so this method is now
        explicitly yard-scoped. Use get_all_gatevision_records() for gate data.
        """
        query = """
        SELECT id, licence_plate_trailer, latitude, longitude, speed, barrier,
               confidence, image_path, camera_id, video_path, frame_number,
               track_id, timestamp, created_on, is_processed,
               assigned_spot_id, assigned_spot_name, assigned_distance_ft, processed_comment
        FROM yardvision_records
        WHERE 1=1
        """
        params: List = []
        if is_processed is not None:
            query += " AND is_processed = ?"
            params.append(1 if is_processed else 0)
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        query += " ORDER BY created_on DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                records = []
                for row in rows:
                    rec = {k: row[k] for k in row.keys()}
                    rec['is_processed'] = bool(rec.get('is_processed'))
                    records.append(rec)
                return records
            except Exception as e:
                print(f"[VideoFrameDB] Error fetching records: {e}")
                return []
            finally:
                conn.close()

    # ------------------------------------------------------------ gate reads

    def get_gatevision_fused_records_pending_upload(
        self,
        limit: int = 50,
        cutoff_seconds: int = 5,
    ) -> List[Dict]:
        """Gate events ready for Prosper upload.

        Returns rows from gatevision_records that:
          - are real gate events (event_type in arrival/departure/pass — excludes 'candidate')
          - have aged past `cutoff_seconds` so concurrent inserts settle.

        Includes both source='live' (real cameras) and source='test' (replay through the
        GateVision AI tab buttons). The Prosper gate-events uploader handles both.

        The returned dict shape stays back-compatible with the prior LIKE-based query —
        callers like prosper_gate_upload still inspect a `video_path` field, so we
        synthesise the legacy string from the structured columns.
        """
        if limit <= 0:
            return []
        query = """
        SELECT id, licence_plate_trailer, latitude, longitude, speed,
               confidence, image_path, camera_id, frame_number, track_id,
               timestamp, created_on,
               gate_id, event_type, source, test_video_stem
        FROM gatevision_records
        WHERE event_type IN ('gate_arrival', 'gate_departure', 'gate_pass')
          AND datetime(created_on) < datetime('now', '-' || ? || ' seconds')
        ORDER BY created_on ASC
        LIMIT ?
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, [cutoff_seconds, limit])
                records = []
                for row in cursor.fetchall():
                    src = row["source"] or "live"
                    et = row["event_type"] or "gate_pass"
                    if src == "test":
                        legacy_vp = f"gatevision:test-{row['test_video_stem'] or 'unknown'}:{et}"
                    else:
                        legacy_vp = f"gatevision:{row['gate_id'] or 'gate-1'}:{et}"
                    rec = {
                        "id": row["id"],
                        "licence_plate_trailer": row["licence_plate_trailer"],
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                        "speed": row["speed"],
                        "barrier": None,  # gate table doesn't carry barrier; legacy callers tolerate None
                        "confidence": row["confidence"],
                        "image_path": row["image_path"],
                        "camera_id": row["camera_id"],
                        "video_path": legacy_vp,  # synthesised for back-compat with prosper_gate_upload
                        "frame_number": row["frame_number"],
                        "track_id": row["track_id"],
                        "timestamp": row["timestamp"],
                        "created_on": row["created_on"],
                        "is_processed": False,
                        "assigned_spot_id": None,
                        "assigned_spot_name": None,
                        "assigned_distance_ft": None,
                        "processed_comment": None,
                        # New structured fields (prefer these for new consumers)
                        "gate_id": row["gate_id"],
                        "event_type": et,
                        "source": src,
                        "test_video_stem": row["test_video_stem"],
                    }
                    records.append(rec)
                return records
            except Exception as e:
                print(f"[VideoFrameDB] Error fetching gatevision Prosper queue: {e}")
                return []
            finally:
                conn.close()

    def get_all_gatevision_records(
        self,
        limit: int = 200,
        offset: int = 0,
        source: Optional[str] = None,
    ) -> List[Dict]:
        """Browse-style read for the GateVision tab UI / debugging."""
        query = "SELECT * FROM gatevision_records WHERE 1=1"
        params: List = []
        if source in ("live", "test"):
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY created_on DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
            except Exception as e:
                print(f"[VideoFrameDB] Error fetching gatevision records: {e}")
                return []
            finally:
                conn.close()

    # ------------------------------------------------------------- statistics

    def get_statistics(self) -> Dict:
        """Aggregate stats across both pipelines.

        Keeps the legacy keys (total/unprocessed/processed) referring to YardVision
        (since that's the only pipeline with an is_processed flow), and adds a
        `gate_total` for completeness.
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        COUNT(*) AS total,
                        SUM(CASE WHEN is_processed = 0 THEN 1 ELSE 0 END) AS unprocessed,
                        SUM(CASE WHEN is_processed = 1 THEN 1 ELSE 0 END) AS processed
                    FROM yardvision_records
                """)
                yard = cursor.fetchone()
                cursor.execute("SELECT COUNT(*) AS total FROM gatevision_records")
                gate = cursor.fetchone()
                yard_total = yard["total"] if yard else 0
                gate_total = gate["total"] if gate else 0
                return {
                    "total": (yard_total or 0) + (gate_total or 0),
                    "unprocessed": yard["unprocessed"] if yard else 0,
                    "processed": yard["processed"] if yard else 0,
                    "yard_total": yard_total or 0,
                    "gate_total": gate_total or 0,
                }
            except Exception as e:
                print(f"[VideoFrameDB] Error getting statistics: {e}")
                return {"total": 0, "unprocessed": 0, "processed": 0, "yard_total": 0, "gate_total": 0}
            finally:
                conn.close()

    # --------------------------------------------------------------- deletes

    def delete_yardvision_by_ids(self, ids: List[int]) -> int:
        """Delete yardvision rows after successful upload."""
        return self._delete_by_ids("yardvision_records", ids)

    def delete_gatevision_by_ids(self, ids: List[int]) -> int:
        """Delete gatevision rows after successful upload."""
        return self._delete_by_ids("gatevision_records", ids)

    def delete_records_by_ids(self, ids: List[int]) -> int:
        """Deprecated: ID is no longer unique across pipelines. Kept as a thin shim that
        deletes from BOTH tables for any matching id. New code should call
        delete_yardvision_by_ids / delete_gatevision_by_ids with already-partitioned IDs.
        """
        if not ids:
            return 0
        return (
            self._delete_by_ids("yardvision_records", ids)
            + self._delete_by_ids("gatevision_records", ids)
        )

    def _delete_by_ids(self, table: str, ids: List[int]) -> int:
        if not ids:
            return 0
        if table not in ("yardvision_records", "gatevision_records"):
            raise ValueError(f"Unknown table: {table}")
        placeholders = ",".join("?" * len(ids))
        sql = f"DELETE FROM {table} WHERE id IN ({placeholders})"
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(sql, ids)
                deleted = cursor.rowcount
                conn.commit()
                return deleted
            except Exception as e:
                print(f"[VideoFrameDB] Error deleting from {table}: {e}")
                conn.rollback()
                return 0
            finally:
                conn.close()
