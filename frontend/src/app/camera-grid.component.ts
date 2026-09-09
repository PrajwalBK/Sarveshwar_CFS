import { Component, input, output } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { Camera, CameraSource, PlaybackAction, UploadState } from './gate-api.service';

@Component({
  selector: 'gate-camera-grid', standalone: true, imports: [DecimalPipe],
  template: `
  <div class="camera-grid">
    @for (camera of cameras(); track camera.id; let index = $index) {
      <article class="camera-card" [attr.aria-label]="camera.name">
        <div class="card-heading"><span class="camera-number">0{{ index + 1 }}</span><div class="camera-name"><strong>{{ camera.name }}</strong><small>{{ camera.gate_id }} · {{ camera.direction }}</small></div>
          <span class="badge" [class.good]="camera.status === 'ONLINE'" [class.warn]="camera.status === 'RECONNECTING'">{{ camera.source_type === 'file' ? camera.playback_status : camera.status }}</span></div>
        <label class="source-selector"><span>Camera source</span><select [disabled]="!!busy()[camera.id] || stale()" (change)="selectSource($event, camera.id)">
          <option value="" disabled [selected]="!camera.active_source_id">Select a camera</option>
          @for (source of cameraSources(); track source.id) {<option [value]="source.id" [selected]="source.id === camera.active_source_id">{{ source.name }}{{ source.configured ? '' : ' — not configured' }}</option>}
        </select></label>
        <div class="camera-frame" (dragover)="$event.preventDefault()" (drop)="drop($event, camera.id)">
          <span class="source-tag">{{ camera.source_type === 'file' ? 'VIDEO FILE' : 'CAMERA INPUT' }}</span>
          @if (camera.has_frame && !stale()) {
            <img [src]="'/api/cameras/' + camera.id + '/stream'" [alt]="camera.name + ' live stream'" (error)="$any($event.target).src = '/api/cameras/' + camera.id + '/frame?t=' + refreshToken()">
          } @else {
            <div class="offline-frame"><svg width="34" height="28" viewBox="0 0 34 28" fill="none" aria-hidden="true"><rect x="2" y="6" width="23" height="18" rx="3" stroke="currentColor" stroke-width="1.5"/><path d="m25 12 7-4v14l-7-4M9 6V3h9v3" stroke="currentColor" stroke-width="1.5"/></svg><strong>{{ stale() ? 'Backend disconnected' : 'No camera connected' }}</strong><small>Drop a recorded video here to test this view</small></div>
          }
        </div>
        <div class="video-input">
          <div class="file-row"><span class="file-name" [title]="camera.video?.filename || ''">{{ camera.source_type === 'file' ? (camera.video?.filename || 'Configured video file') : 'No video selected' }}</span>
            <label class="upload-button" [class.disabled]="uploads()[camera.id]?.busy"><input type="file" [attr.aria-label]="'Add video for ' + camera.name" accept=".mp4,.avi,.mov,.mkv,.webm,.m4v" [disabled]="uploads()[camera.id]?.busy || stale()" (change)="choose($event, camera.id)">{{ uploads()[camera.id]?.busy ? 'Uploading…' : camera.video?.active ? 'Replace video' : '+ Add video' }}</label></div>
          @if (uploads()[camera.id]; as state) {
            <div class="upload-status" [class.error-text]="state.error" role="status">{{ state.message }}</div>
            @if (state.busy) {<progress [value]="state.progress" max="100" aria-label="Upload progress"></progress>}
          }
          @if (camera.source_type === 'file') {
            <div class="playback-timeline"><progress [value]="camera.video_position_seconds" [max]="camera.video_duration_seconds || 1" aria-label="Video playback progress"></progress><small>{{ camera.video_position_seconds | number:'1.0-0' }}s / {{ camera.video_duration_seconds === null ? '—' : (camera.video_duration_seconds | number:'1.0-0') + 's' }}</small></div>
            <div class="playback-buttons"><button class="primary" (click)="control.emit({id: camera.id, action: camera.playback_status === 'PLAYING' ? 'pause' : 'play'})" [disabled]="!!busy()[camera.id] || stale()">{{ camera.playback_status === 'PLAYING' ? 'Ⅱ Pause' : '▶ Play' }}</button><button (click)="control.emit({id: camera.id, action: 'restart'})" [disabled]="!!busy()[camera.id] || stale()">Restart</button><button class="text-button" (click)="control.emit({id: camera.id, action: 'restore-camera'})" [disabled]="!!busy()[camera.id] || stale()">Use camera</button></div>
          }
        </div>
        <dl class="camera-stats"><div><dt>CAPTURE</dt><dd>{{ camera.capture_fps | number:'1.1-1' }} <span>fps</span></dd></div><div><dt>INFERENCE</dt><dd>{{ inferenceFps()[camera.id] ?? 0 | number:'1.1-1' }} <span>fps</span></dd></div><div><dt>DROPPED</dt><dd>{{ camera.dropped_frames }}</dd></div></dl>
      </article>
    } @empty { <p class="empty">Loading the four input slots…</p> }
  </div>`
})
export class CameraGridComponent {
  cameras = input<Camera[]>([]);
  cameraSources = input<CameraSource[]>([]);
  inferenceFps = input<Partial<Record<string, number>>>({});
  refreshToken = input(0);
  stale = input(false);
  uploads = input<Partial<Record<string, UploadState>>>({});
  busy = input<Record<string, boolean>>({});
  upload = output<{id: string; file: File}>();
  control = output<{id: string; action: PlaybackAction}>();
  sourceChange = output<{id: string; sourceId: string}>();
  selectSource(event: Event, id: string) {
    this.sourceChange.emit({id, sourceId: (event.target as HTMLSelectElement).value});
  }
  choose(event: Event, id: string) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) this.upload.emit({id, file});
    input.value = '';
  }
  drop(event: DragEvent, id: string) {
    event.preventDefault();
    if (this.stale() || this.uploads()[id]?.busy) return;
    const file = event.dataTransfer?.files[0];
    if (file) this.upload.emit({id, file});
  }
}
