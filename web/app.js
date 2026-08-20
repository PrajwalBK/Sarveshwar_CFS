// GateVision AI Dashboard JavaScript

const API_BASE = '';
function apiUrl(path) {
    const base = (typeof API_BASE === 'string' && API_BASE) ? API_BASE.replace(/\/$/, '') : '';
    return base + (path.startsWith('/') ? path : '/' + path);
}

const METRICS_URL = apiUrl('/metrics.json');
const REFRESH_INTERVAL = 2000;

let metricsInterval;
let gateVisionInterval;
let resultsInterval;
let currentVideoId = null;

// Initialize dashboard on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    loadMetrics();
    setupGateVideoProcessing();
    setupMultiCameraControls();
    setupGateSyncControls();
    loadGateVisionStatus();
    startAllCameraStreams();

    metricsInterval = setInterval(loadMetrics, REFRESH_INTERVAL);
    gateVisionInterval = setInterval(loadGateVisionStatus, REFRESH_INTERVAL);
});

// ========== METRICS & CONNECTION ==========

async function loadMetrics() {
    try {
        const response = await fetch(METRICS_URL);
        if (response.ok) {
            updateStatus(true);
        } else {
            updateStatus(false);
        }
    } catch (error) {
        updateStatus(false);
    }
}

function updateStatus(connected) {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    if (!statusDot || !statusText) return;

    if (connected) {
        statusDot.classList.remove('disconnected');
        statusText.textContent = 'Connected';
    } else {
        statusDot.classList.add('disconnected');
        statusText.textContent = 'Disconnected';
    }
}

// ========== GATE VISION STATUS & EVENTS ==========

async function loadGateVisionStatus() {
    try {
        const res = await fetch(apiUrl('/api/gatevision/status'));
        if (!res.ok) return;
        const data = await res.json();
        if (!data || !data.success) return;

        // Update counts
        const st = data.stats || {};
        const elEmitted = document.getElementById('gateEventsEmitted');
        const elConfirmed = document.getElementById('gateEventsConfirmed');
        const elReview = document.getElementById('gateEventsNeedsReview');
        const elLast = document.getElementById('gateLastEventAt');

        if (elEmitted) elEmitted.textContent = st.events_emitted ?? 0;
        if (elConfirmed) elConfirmed.textContent = st.confirmed_events ?? 0;
        if (elReview) elReview.textContent = st.needs_review_events ?? 0;
        if (elLast) elLast.textContent = st.last_event_ts ? new Date(st.last_event_ts).toLocaleTimeString() : '—';

        // Update Prosper upload status
        const up = data.upload || {};
        const uEnabled = document.getElementById('gateUploadEnabled');
        const uThread = document.getElementById('gateUploadThreadAlive');
        const uUploading = document.getElementById('gateUploadIsUploading');
        const uLastRun = document.getElementById('gateUploadLastRun');
        const uLastRes = document.getElementById('gateUploadLastResult');
        const uTotal = document.getElementById('gateUploadTotal');

        if (uEnabled) uEnabled.textContent = up.enabled ? 'Yes' : 'No';
        if (uThread) uThread.textContent = up.thread_alive ? 'Running' : 'Stopped';
        if (uUploading) uUploading.textContent = up.is_uploading ? 'Syncing...' : 'Idle';
        if (uLastRun) uLastRun.textContent = up.last_run ? new Date(up.last_run).toLocaleTimeString() : '—';
        if (uTotal) uTotal.textContent = up.total_uploaded ?? 0;

        if (uLastRes) {
            const resVal = up.last_result || '—';
            uLastRes.textContent = resVal.toUpperCase();
            uLastRes.className = 'upload-result-badge ' + (resVal === 'success' ? 'success' : resVal === 'failed' ? 'error' : '');
        }

        // Render recent gate pass events table
        if (Array.isArray(data.events)) {
            renderGateEvents(data.events);
        }
    } catch (e) {
        console.debug('GateVision status poll notice:', e);
    }
}

function renderGateEvents(events) {
    const tbody = document.getElementById('gateEventsTableBody');
    if (!tbody) return;

    if (!events.length) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #888;">No gate pass events logged yet</td></tr>';
        return;
    }

    const rows = events.slice().reverse().slice(0, 30).map((e) => {
        const conf = typeof e.conf === 'number' ? (e.conf * 100).toFixed(1) + '%' : '—';
        const ts = e.ts_iso ? new Date(e.ts_iso).toLocaleString() : '—';
        const containerId = e.trailer_id || e.container_number || 'UNKNOWN';
        const status = e.status || 'PROCESSED';

        return `<tr>
            <td>${ts}</td>
            <td><strong>${e.gate_id || 'GATE-01'}</strong></td>
            <td>${e.direction || 'INBOUND'}</td>
            <td><code style="font-weight: bold; font-size: 1.05em; color: #4dabf7;">${containerId}</code></td>
            <td><span class="status-badge ${status.toLowerCase()}">${status}</span></td>
            <td>${conf}</td>
        </tr>`;
    });

    tbody.innerHTML = rows.join('');
}

// ========== GATE VIDEO PROCESSING ==========

function setupGateVideoProcessing() {
    const fileInput = document.getElementById('gateVideoFileInput');
    const selectBtn = document.getElementById('gateSelectVideoBtn');
    const serverSelect = document.getElementById('gateServerVideoSelect');
    const processBtn = document.getElementById('gateProcessVideoBtn');
    const stopBtn = document.getElementById('gateStopProcessingBtn');
    const statusEl = document.getElementById('gateProcessingStatus');

    if (!selectBtn || !processBtn) return;

    function updateGateStatus(msg, type) {
        if (!statusEl) return;
        statusEl.textContent = msg;
        statusEl.className = 'processing-status ' + (type || '');
    }

    // Load available server recordings
    async function loadServerRecordings() {
        if (!serverSelect) return;
        try {
            const res = await fetch(apiUrl('/api/gatevision/test-recordings/list'));
            if (!res.ok) return;
            const data = await res.json();
            if (data.success && Array.isArray(data.videos) && data.videos.length > 0) {
                serverSelect.innerHTML = '<option value="">— Select a server video (' + data.videos.length + ' available) —</option>' +
                    data.videos.map(v => `<option value="${v.path}">${v.name} (${(v.size / (1024*1024)).toFixed(1)} MB)</option>`).join('');
                
                // Auto-select first video if not chosen
                if (!currentVideoId && data.videos.length > 0) {
                    serverSelect.value = data.videos[0].path;
                    currentVideoId = data.videos[0].path;
                    processBtn.disabled = false;
                    updateGateStatus(`Loaded server video: ${data.videos[0].name}. Click Process Gate Video to start.`, 'idle');
                }
            } else {
                serverSelect.innerHTML = '<option value="">— No recordings found on server —</option>';
            }
        } catch (e) {
            console.debug('Failed to load server recordings list:', e);
        }
    }

    loadServerRecordings();

    if (serverSelect) {
        serverSelect.addEventListener('change', () => {
            if (serverSelect.value) {
                currentVideoId = serverSelect.value;
                processBtn.disabled = false;
                updateGateStatus(`Selected video: ${serverSelect.options[serverSelect.selectedIndex].text}. Click Process Gate Video.`, 'idle');
            } else {
                processBtn.disabled = true;
            }
        });
    }

    selectBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
            selectBtn.disabled = true;
            selectBtn.textContent = 'Uploading...';

            const formData = new FormData();
            formData.append('video', file);
            formData.append('detection_mode', 'gate');

            const response = await fetch(apiUrl('/api/upload-video'), {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || 'Upload failed');
            }

            const data = await response.json();
            currentVideoId = data.video_id || data.video_path;

            selectBtn.disabled = false;
            selectBtn.textContent = '📁 Upload Video File';
            updateGateStatus(`Gate video uploaded (${file.name}). Click Process Gate Video to start.`, 'success');
            processBtn.disabled = false;
            loadServerRecordings();
        } catch (error) {
            console.error('Error uploading gate video:', error);
            updateGateStatus('Upload failed: ' + error.message, 'error');
            selectBtn.disabled = false;
            selectBtn.textContent = '📁 Upload Video File';
        }
    });

    // 4-Camera Multi-Stream Management
    const GATE_CAMERAS = [
        { id: 'gate-in-01', name: 'Gate IN - Front Camera', role: 'gate_in_front', direction: 'INBOUND' },
        { id: 'gate-in-02', name: 'Gate IN - Rear Camera', role: 'gate_in_rear', direction: 'INBOUND' },
        { id: 'gate-out-01', name: 'Gate OUT - Front Camera', role: 'gate_out_front', direction: 'OUTBOUND' },
        { id: 'gate-out-02', name: 'Gate OUT - Rear Camera', role: 'gate_out_rear', direction: 'OUTBOUND' }
    ];

    window.GATE_CAMERAS = GATE_CAMERAS;

    // Connect individual camera button and role change bindings
    GATE_CAMERAS.forEach(cam => {
        const btn = document.getElementById(`btn-connect-${cam.id}`);
        if (btn) {
            btn.addEventListener('click', () => connectCameraStream(cam.id));
        }

        const roleSel = document.getElementById(`role-select-${cam.id}`);
        if (roleSel) {
            roleSel.addEventListener('change', () => {
                updateRoleBadge(cam.id);
            });
        }
    });

    // Auto-Scan LAN Cameras Button
    const scanAllBtn = document.getElementById('gateScanAllCamerasBtn');
    if (scanAllBtn) {
        scanAllBtn.addEventListener('click', async () => {
            try {
                scanAllBtn.disabled = true;
                scanAllBtn.textContent = '⏳ Scanning LAN...';
                updateGateStatus('Scanning local Ethernet for CP Plus / ONVIF IP cameras...', 'processing');

                const res = await fetch(apiUrl('/api/gatevision/camera/scan'), { method: 'POST' });
                const data = await res.json();
                
                scanAllBtn.disabled = false;
                scanAllBtn.textContent = '🔍 Auto-Scan LAN';

                if (data.success && Array.isArray(data.cameras) && data.cameras.length > 0) {
                    data.cameras.forEach((cam, idx) => {
                        if (idx < GATE_CAMERAS.length) {
                            const cid = GATE_CAMERAS[idx].id;
                            const input = document.getElementById(`rtsp-input-${cid}`);
                            if (input) {
                                input.value = `rtsp://admin:admin@${cam.ip}:554/cam/realmonitor?channel=1&subtype=1`;
                            }
                        }
                    });
                    updateGateStatus(`✓ Discovered ${data.cameras.length} camera(s) on network! Ready to connect.`, 'success');
                } else {
                    updateGateStatus('Scan complete. Default IPs configured for all 4 channels.', 'idle');
                }
            } catch (e) {
                scanAllBtn.disabled = false;
                scanAllBtn.textContent = '🔍 Auto-Scan LAN';
                updateGateStatus('Scan notice: ' + e.message, 'warning');
            }
        });
    }

    // Connect All 4 Cameras Button
    const connectAllBtn = document.getElementById('gateConnectAllBtn');
    if (connectAllBtn) {
        connectAllBtn.addEventListener('click', async () => {
            connectAllBtn.disabled = true;
            connectAllBtn.textContent = '⏳ Connecting All...';
            updateGateStatus('Connecting all 4 Gate AI Streams (2 Inbound + 2 Outbound)...', 'processing');

            for (const cam of GATE_CAMERAS) {
                await connectCameraStream(cam.id, false);
            }

            connectAllBtn.disabled = false;
            connectAllBtn.textContent = '▶ Connect All 4 Cameras';
            updateGateStatus('✓ All 4 Gate AI Streams Active & Connected!', 'success');
            startAllCameraStreams();
            startResultsPolling();
        });
    }

    // Stop Streams Button
    const stopBtn = document.getElementById('gateStopProcessingBtn');
    if (stopBtn) {
        stopBtn.addEventListener('click', async () => {
            try {
                await fetch(apiUrl('/api/stop-processing'), { method: 'POST' });
                GATE_CAMERAS.forEach(cam => {
                    const badge = document.getElementById(`status-badge-${cam.id}`);
                    if (badge) {
                        badge.textContent = 'IDLE';
                        badge.style.borderColor = '#64748b';
                        badge.style.color = '#94a3b8';
                    }
                });
                updateGateStatus('All camera processing stopped.', 'idle');
            } catch (e) {
                console.error(e);
            }
        });
    }

    processBtn.addEventListener('click', async () => {
        if (!currentVideoId) {
            updateGateStatus('Please select or upload a video file first.', 'warning');
            return;
        }

        try {
            processBtn.disabled = true;
            stopBtn.disabled = false;
            updateGateStatus('Processing Gate Video (Running custom YOLOv8 container detections)...', 'processing');

            const response = await fetch(apiUrl(`/api/process-video/${encodeURIComponent(currentVideoId)}?detection_mode=gate`), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_id: currentVideoId,
                    video_path: currentVideoId,
                    detect_every_n: 5,
                    detection_mode: 'gate'
                })
            });

            const data = await response.json().catch(() => ({}));
            if (!response.ok && !data.already_running) {
                throw new Error(data.error || 'Processing failed to start');
            }

            updateGateStatus('Gate Video Processing is running! Live detections streaming below...', 'success');
            startGateProcessedVideoStream();
            startResultsPolling();
        } catch (error) {
            console.error('Error starting gate processing:', error);
            updateGateStatus('Notice: ' + error.message, 'error');
            processBtn.disabled = false;
            stopBtn.disabled = true;
        }
    });

    stopBtn.addEventListener('click', async () => {
        try {
            await fetch(apiUrl('/api/stop-processing'), { method: 'POST' });
            processBtn.disabled = false;
            stopBtn.disabled = true;
            updateGateStatus('Processing stopped.', 'idle');
        } catch (error) {
            console.error('Error stopping gate processing:', error);
        }
    });
}

function startGateProcessedVideoStream() {
    let img = document.getElementById('gateProcessedVideoStream');
    if (!img) {
        const container = document.getElementById('gateProcessedVideoContainer');
        if (!container) return;
        img = document.createElement('img');
        img.id = 'gateProcessedVideoStream';
        img.className = 'processed-video-stream';
        img.style.width = '100%';
        img.style.maxHeight = '520px';
        img.style.objectFit = 'contain';
        img.style.borderRadius = '8px';
        img.alt = 'Live Gate AI Detections';
        container.appendChild(img);
    }
    img.onerror = () => {
        setTimeout(() => {
            img.src = apiUrl('/api/processed-video-stream?t=') + new Date().getTime();
        }, 2000);
    };
    img.src = apiUrl('/api/processed-video-stream?t=') + new Date().getTime();
}

function startResultsPolling() {
    if (resultsInterval) clearInterval(resultsInterval);

    resultsInterval = setInterval(async () => {
        try {
            const statusRes = await fetch(apiUrl('/api/processing-status'));
            if (statusRes.ok) {
                const status = await statusRes.json();
                if (status.status === 'completed') {
                    const processBtn = document.getElementById('gateProcessVideoBtn');
                    const stopBtn = document.getElementById('gateStopProcessingBtn');
                    if (processBtn) processBtn.disabled = false;
                    if (stopBtn) stopBtn.disabled = true;
                    clearInterval(resultsInterval);
                    loadGateVisionStatus();
                }
            }
        } catch (e) {
            console.debug('Polling error:', e);
        }
    }, 1500);
}

// ========== GATE PROSPER SYNC CONTROLS ==========

function setupGateSyncControls() {
    const btn = document.getElementById('gateUploadNowBtn');
    if (!btn) return;

    btn.addEventListener('click', async () => {
        btn.disabled = true;
        btn.textContent = 'Syncing...';

        try {
            const res = await fetch(apiUrl('/api/gatevision/upload-now'), { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                btn.textContent = '✓ Synced!';
                loadGateVisionStatus();
            } else {
                btn.textContent = 'Sync Notice';
            }
        } catch (e) {
            btn.textContent = 'Error';
        }

        setTimeout(() => {
            btn.disabled = false;
            btn.textContent = 'Sync Pending to Prosper';
        }, 2000);
    });
}

// ========== 4-CAMERA MULTI-STREAM CONNECTORS ==========

// ========== 4-CAMERA MULTI-STREAM CONNECTORS ==========

function updateRoleBadge(cameraId) {
    const roleSelect = document.getElementById(`role-select-${cameraId}`);
    const dirBadge = document.getElementById(`dir-badge-${cameraId}`);
    const nameLabel = document.getElementById(`name-label-${cameraId}`);
    const card = document.getElementById(`card-${cameraId}`);

    if (!roleSelect) return;
    const role = roleSelect.value;
    const isInbound = role.includes('in') || role.includes('entry');

    if (dirBadge) {
        dirBadge.textContent = isInbound ? 'INBOUND' : 'OUTBOUND';
        dirBadge.style.background = isInbound ? '#15803d' : '#0284c7';
    }

    if (nameLabel) {
        const isRear = role.includes('rear');
        nameLabel.textContent = `${isInbound ? 'Gate IN' : 'Gate OUT'} - ${isRear ? 'Rear' : 'Front'} Camera`;
        nameLabel.style.color = isInbound ? '#4ade80' : '#38bdf8';
    }

    if (card) {
        card.style.borderColor = isInbound ? 'rgba(34, 197, 94, 0.3)' : 'rgba(56, 189, 248, 0.3)';
    }
}

async function connectCameraStream(cameraId, updateUi = true) {
    const input = document.getElementById(`rtsp-input-${cameraId}`);
    const roleSelect = document.getElementById(`role-select-${cameraId}`);
    const rtspUrl = (input ? input.value : '').trim();
    const role = roleSelect ? roleSelect.value : 'gate_in_front';
    const isInbound = role.includes('in') || role.includes('entry');
    const direction = isInbound ? 'INBOUND' : 'OUTBOUND';
    const gateId = isInbound ? 'gate-in' : 'gate-out';

    const badge = document.getElementById(`status-badge-${cameraId}`);
    const statusEl = document.getElementById('gateProcessingStatus');

    if (!rtspUrl) return;

    updateRoleBadge(cameraId);

    if (badge) {
        badge.textContent = 'CONNECTING...';
        badge.style.borderColor = '#f59e0b';
        badge.style.color = '#f59e0b';
    }

    try {
        const res = await fetch(apiUrl('/api/gatevision/camera/connect'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                camera_id: cameraId,
                rtsp_url: rtspUrl,
                role: role,
                direction: direction,
                gate_id: gateId
            })
        });

        // Also refresh stream image immediately
        setTimeout(() => {
            const img = document.getElementById(`stream-${cameraId}`);
            if (img) {
                const ts = new Date().getTime();
                img.src = apiUrl(`/stream/${cameraId}?t=${ts}`);
                img.onerror = () => {
                    setTimeout(() => {
                        img.src = apiUrl(`/stream/${cameraId}?t=`) + new Date().getTime();
                    }, 2000);
                };
            }
        }, 500);

        if (badge) {
            badge.textContent = '● LIVE';
            badge.style.borderColor = isInbound ? '#22c55e' : '#38bdf8';
            badge.style.color = isInbound ? '#4ade80' : '#38bdf8';
        }

        if (updateUi && statusEl) {
            statusEl.textContent = `✓ Connected ${cameraId} (${direction} / ${role})! Streaming live...`;
            statusEl.className = 'processing-status success';
        }
    } catch (e) {
        if (badge) {
            badge.textContent = 'OFFLINE';
            badge.style.borderColor = '#ef4444';
            badge.style.color = '#ef4444';
        }
        if (updateUi && statusEl) {
            statusEl.textContent = `Notice on ${cameraId}: ${e.message}`;
            statusEl.className = 'processing-status error';
        }
    }
}

function startAllCameraStreams() {
    const ts = new Date().getTime();
    const cams = window.GATE_CAMERAS || [
        { id: 'gate-in-01' }, { id: 'gate-in-02' },
        { id: 'gate-out-01' }, { id: 'gate-out-02' }
    ];

    cams.forEach(cam => {
        updateRoleBadge(cam.id);
        const img = document.getElementById(`stream-${cam.id}`);
        if (img) {
            img.src = apiUrl(`/stream/${cam.id}?t=${ts}`);
            img.onerror = () => {
                setTimeout(() => {
                    img.src = apiUrl(`/stream/${cam.id}?t=`) + new Date().getTime();
                }, 3000);
            };
        }
    });
}
