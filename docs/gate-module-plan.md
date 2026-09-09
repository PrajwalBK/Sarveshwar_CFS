# Gate Module assessment and implementation plan

Assessment date: 2026-09-08. Scope: four fixed gate cameras and a development-capable POC.

## A. Historical architecture at assessment time

At assessment time, the root `app/main_trt_demo.py` orchestrated per-camera threads, model loading, offline processing, SQLite, publishing and Prosper uploads. `app/metrics_server.py` was a Flask API and static-file server. That root `app/` implementation has since been retired and removed after the independent FastAPI/Angular Gate Module was verified. `web/` remains only as a legacy folder and is not used by the active module.

`app/rtsp.py` captures RTSP/USB with OpenCV and optional Jetson GStreamer. The detector factory chooses hardcoded candidate checkpoint paths. `ByteTrackWrapper` implements local IoU/Kalman association. OCR includes EasyOCR, transformer/VLM and TensorRT implementations. Gate processing buffers crops for asynchronous OCR; fusion groups observations by gate/time. SQLite contains `yardvision_records` and `gatevision_records`. Docker Compose describes observability, messaging and Timescale ingest, but the referenced `services/ingest` source directory is absent. Existing uploads can remove successfully synchronized SQLite rows.

The historical worktree changes under root `app/` were intentionally removed with that application. Other legacy folders and unrelated worktree changes remain untouched.

## B. Gap analysis

| Requirement | Existing capability | Work needed |
|---|---|---|
| FastAPI / Angular / MySQL | Flask / static JS / SQLite | Separate deployable Gate service and Angular workspace |
| Camera isolation | Per-camera threads | Explicit manager, reconnect/backoff, bounded latest-frame buffers, health |
| Model configuration | YOLO/TensorRT wrappers | Interface, configurable path/backend/classes; no hidden checkpoint fallback |
| Container validation | Generic plate filtering | ISO 6346 structure/check digit; raw OCR and validation evidence |
| Event association | Gate/time correlation | Track confirmation and validated-ID association across views; restart-aware dedup |
| Direction | Name/role substring heuristics | Explicit lane direction or calibrated line crossing; unknown by default |
| Snapshots | Crops saved before OCR | Context snapshots only when an event is persisted |
| API / persistence | Large coupled modules | Small routes, repository, normalized relational schema |
| Tests / measurement | Scattered scripts, some unit tests | Hardware-free pipeline tests, MySQL integration option, measured metrics |

The previous `fusion.py` can merge distinct vehicles within its time window. Camera names containing `front` can also vote inbound even for an outbound lane. Those heuristics are not carried into the new service.

## C. Proposed architecture

Four capture workers -> one latest-frame slot per camera -> fair inference scheduler -> detector interface -> per-camera temporal tracker -> bounded OCR queue -> OCR interface and container validator -> event manager -> SQLAlchemy/MySQL transaction and event snapshot -> FastAPI -> Angular.

One shared accelerator lock serializes YOLO and OCR execution to bound GPU pressure. Capture continues independently during OCR. Frames superseded before processing are counted. File sources support paced replay. The API remains responsive when a camera/model fails. A single service process owns cameras and model workers; do not launch multiple Uvicorn workers.

An event requires repeated detection and OCR evidence. Valid IDs associate only within the same configured gate/lane and compatible direction, inside a configurable dedup window. Different valid IDs never merge solely because timestamps overlap. Invalid reads become review events attached to a camera/track; they cannot safely merge across views. Unknown direction remains UNKNOWN. This is temporal association for a controlled POC, not a guarantee of identity through long occlusions.

## D. Files and incremental changes

1. `backend/app/config`, `database`, `database/repositories`: typed settings, camera YAML with environment references, models and explicit schema initialization.
2. `backend/app/camera`: stream abstraction, capture workers and manager.
3. `backend/app/detection`, `ocr`: typed outputs, YOLO adapter, separate tracker, EasyOCR adapter and ISO validation.
4. `backend/app/events`, `snapshots`, `runtime.py`: direction, confirmation/dedup, bounded queues and event persistence.
5. `backend/app/api`, `main.py`: read APIs, camera tests, lifecycle, health and metrics.
6. `frontend/`: standalone Angular overview, cameras, event table/details and snapshots, API polling with failure states.
7. `backend/tests`, deployment configuration and runbook: simulation, regression coverage and site acceptance procedure.

Reuse trained local checkpoints through configuration, real footage, calibration knowledge, deployment/observability assets and existing tools. Keep legacy OCR/Prosper adapters available for later explicit integration; importing their orchestration would bring side effects and unnecessary GPU dependencies into the new API. No automatic legacy data migration, upload or deletion.

## E. Database

MySQL with SQLAlchemy; explicit `python -m app.database.initialize` initializes a new schema and records its version. No legacy tables are dropped. Future model changes require an explicit versioned migration.

- `cameras`: ID, name, lane/gate, source environment key, enabled flag, direction configuration, UTC timestamps. Never store RTSP credentials.
- `gate_events`: UUID, gate/lane, time, container number if valid, type ENTRY/EXIT/UNKNOWN, status, deterministic track-origin key.
- `detections`: UUID, event FK, camera FK, track ID, class, confidence, bbox and capture timestamp. Persist event evidence; recent frame detections stay in a bounded memory cache.
- `ocr_results`: detection FK, raw/normalized text, confidence, format/check-digit flags, validation status.
- `snapshots`: event and camera FKs, local relative image reference, capture timestamp.
- `system_logs`: timestamp, level, event name, non-secret context, event FK when applicable.

Index events by time and `(gate_id, container_number, event_type, timestamp)` for dedup queries. Make camera/run/track origin unique for idempotent retries. Event evidence and snapshot metadata commit atomically. Filesystem and SQL cannot share a transaction: remove a newly written image on rollback; document orphan recovery for a process crash.

## F. APIs

Recorded-video testing is implemented through `POST /api/cameras/{id}/video`, `POST /api/cameras/{id}/playback`, `POST /api/videos/play-all` and `POST /api/processing/start`. Each slot independently accepts a validated, size-limited upload and supports preview/play/pause/restart or restoration of its original source. Assignments persist across restarts. Preview/playback remains available when MySQL is unavailable; AI event storage still requires MySQL. Uploaded clips are retained until operator maintenance. Source-generation checks prevent queued OCR from a replaced recording being saved against its replacement.

Physical PoE cameras are registered by friendly ID/name and environment-variable reference. `GET /api/cameras/sources/available` supplies the sanitized dropdown list and `POST /api/cameras/{slot}/source` switches a logical slot. Per-slot choices persist locally; switching stops the old worker, deactivates any manual video and resets tracking so observations cannot cross source boundaries. Connectivity is reported by the worker rather than inferred from the presence of an RTSP setting.

The Angular dashboard includes CPU/RAM and measured GPU power, temperature, utilization and VRAM. Unsupported GPU readings remain unavailable. Site/admin setup and authentication are deferred by user choice. Cleanup is limited to the new Gate Module; the legacy Flask/static-JavaScript application is preserved.

`GET /api/cameras`, `GET /api/cameras/{id}/status`, `POST /api/cameras/{id}/test`, `GET /api/cameras/{id}/frame`.

`GET /api/gate-events` (pagination, camera/container/type/status filters), `GET /api/gate-events/{id}`, `GET /api/ocr-results`, `GET /api/detections`, `GET /api/snapshots/{id}`.

`GET /api/health`, `GET /api/models`, `GET /api/metrics`. Configuration is declarative through YAML/environment with restart; APIs return sanitized public configuration. Camera tests only use configured sources. Development binds to loopback; site exposure requires the deployment's authenticated reverse proxy/TLS policy. WebRTC is optional and the existing server remains the streaming reference.

## G. AI implementation sequence

Configure four sources and model class aliases (existing training uses names including `Cointainer`, `Feet`, `container_object`; do not assume these mean the same physical object). Load YOLO once from an explicitly selected checkpoint. Ultralytics performs fit/pad preprocessing and returns original-frame boxes. Clip boxes before cropping. Track only inference frames, using elapsed time to expire tracks. Require observed hits before OCR, throttle retries per track and use a bounded queue. OCR a configurable relative region of a container/trailer crop, with whole-container fallback. Normalize text, calculate ISO check digit, accumulate repeated evidence, then create or associate an event. Save a full-frame context snapshot with the selected bbox. Collect capture/inference/OCR/event metrics without promising a fixed FPS.

PyTorch, ONNX and TensorRT are adapter choices; optimized artifacts must be exported/validated for the eventual JetPack and device. Desktop TensorRT engines are not assumed portable. No Jetson libraries are imported by API/domain modules.

## H. Testing and acceptance

Automated tests: settings and secret exclusion; four-camera isolation; read/open failure reconnect and shutdown; paced file EOF; detector output using injected model; class-aware tracking and expiry; crop clipping; OCR cleanup and checksum fixtures; line crossing and UNKNOWN fallback; confirmation, cross-view dedup, conflicting IDs, repeat visits, restart and DB-failure retry; transaction rollback/FKs; API pagination, missing resources, sanitized errors, snapshots and health; full simulated frames-to-event flow. SQLite is only a fast test double. Run the same repository tests against disposable MySQL via `TEST_MYSQL_URL` before site deployment.

Build Angular with strict template checks. Site acceptance: all four cameras concurrently, unplug/replug each, stable resource use over a long run, evidence review for lighting/viewpoint cases, container exact-match rate, invalid-read/review rate, event duplicate/miss rate, direction accuracy, disk growth and measured p50/p95 latency. Test separate train/validation/test footage and unseen site clips. Real camera accuracy, MySQL live operation and Jetson performance remain acceptance gates until measured on those systems.

References: [BIC identification](https://www.bic-code.org/identification-number/), [BIC check digit](https://www.bic-code.org/check-digit-calculator/), [Ultralytics prediction](https://docs.ultralytics.com/modes/predict/), [thread safety](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/), [Angular compatibility](https://angular.dev/reference/versions).
