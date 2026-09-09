# GateVision — Container Gate Module

The active application is the modular GateVision gate system. In deployment, FastAPI serves the compiled Angular UI so one process starts the UI, API, camera workers and AI runtime.

- `backend/`: FastAPI, camera/video workers, AI pipeline, MySQL persistence and operations APIs.
- `frontend/`: Angular operations dashboard.
- `docs/brain.md`: complete architecture and operating flow.

The former root `app/` Python application has been retired and removed. Remaining legacy folders are not part of the active Gate Module.

## One-command startup

From the project root on Windows:

```powershell
.\start.ps1
```

Open `http://127.0.0.1:8001`. Press Ctrl+C in that terminal to stop the complete application. The script uses `backend/.venv`, builds Angular automatically only when the compiled UI is missing, and then runs one FastAPI worker.

To rebuild the UI before starting:

```powershell
.\start.ps1 -BuildFrontend
```

For access from another device during a controlled deployment test:

```powershell
.\start.ps1 -HostAddress 0.0.0.0
```

Do not expose it outside the trusted network until authentication and TLS are installed. MySQL must be configured before schema initialization and AI event processing. Camera selection, manual uploads and previews remain available while the database is unavailable.

`npm start` remains available only for frontend development with hot reload; it is not the deployment startup path.

## Verify

```powershell
cd backend
.\.venv\Scripts\python -m pytest -q
cd ..\frontend
npm run build
```

See [the project brain](docs/brain.md) for camera configuration, source switching, manual-video behavior, AI/event flow, APIs, failure handling and deployment requirements.
