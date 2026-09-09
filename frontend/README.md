# Angular Gate Operations UI

Standalone Angular 21 workspace for the active FastAPI Gate backend. The remaining root `../web` folder is legacy and is not used by this workspace.

Use Node 20.19+, 22.12+, or 24.x compatible with Angular 21 (see [Angular compatibility](https://angular.dev/reference/versions)).

```text
npm ci
npm start
```

Visit `http://127.0.0.1:4200`. The development proxy sends `/api` to `http://127.0.0.1:8001`. Start the backend first, or use `python -m tools.simulate --serve` from `../backend` for explicitly synthetic API/UI test data.

`npm run build` runs strict TypeScript/template checks and creates `dist/gate/browser`. Host those static files behind the same origin as a reverse proxy for `/api`; do not expose RTSP credentials to the browser. No external fonts, chart services or camera connections are needed.

Features: a PoE camera dropdown for every logical slot, plus four independent video uploads/drop zones, upload progress, first-frame previews, Play/Pause/Restart, restore-camera and Play-all controls. Selecting a camera exits video mode for that slot and the choice survives backend restarts. Dropdowns show friendly names and configuration state; RTSP URLs and credentials stay on the backend. The health panel shows database/model status, OCR queue, CPU/RAM and measured GPU power, utilization, temperature and VRAM. Site and admin setup is deferred: no invented site or logged-in identity is shown.

To test now, start both services and choose **Add video** in each camera card. No camera connection, MySQL or AI model is required for video preview/playback. **Enable AI** requires configured models and initialized MySQL. Upload assignments survive backend restarts; replacing clips does not delete earlier uploaded files. See the backend README for upload limits and storage policy.

Paginated/filtered gate events, detection/OCR confidence and snapshot details remain available when event storage is connected. Polling failures mark data stale; event-storage errors do not block uploads. A 503 backend health response is rendered as degraded rather than discarded. Previews are refreshed JPEGs without audio. WebRTC is not a dependency. No authentication is implemented; keep development services on loopback.
