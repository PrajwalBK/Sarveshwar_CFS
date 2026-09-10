import { Component, input, output, signal } from '@angular/core';
import { DatePipe, PercentPipe } from '@angular/common';
import { EventDetail } from './gate-api.service';

@Component({
  selector: 'gate-event-detail',
  standalone: true,
  imports: [DatePipe, PercentPipe],
  template: `
    @if (event(); as item) {
      <div class="detail-page" aria-label="Event details page">
        <div class="detail-header-bar">
          <div class="detail-nav-actions">
            <button class="back-button" (click)="back.emit()">
              <span class="arrow">←</span> Back to Recent gate events
            </button>
            <div class="detail-breadcrumbs">
              <span>Gate events</span>
              <span class="sep">/</span>
              <span class="current">{{ item.container_number || item.id.slice(0, 8) }}</span>
            </div>
          </div>
          <div class="detail-top-actions">
            <a class="action-btn download-btn" [href]="'/api/gate-events/' + item.id + '/json'" [download]="'event_' + (item.container_number || item.id) + '.json'">
              <span class="icon">📥</span> Download event.json
            </a>
            <button class="action-btn json-toggle-btn" [class.active]="showJson()" (click)="toggleJson()">
              <span class="icon">{{ showJson() ? '✕' : '{ }' }}</span> {{ showJson() ? 'Hide raw JSON' : 'View event.json' }}
            </button>
          </div>
        </div>

        <div class="event-hero-card">
          <div class="hero-main">
            <div class="hero-label">CONTAINER IDENTIFICATION & EVIDENCE</div>
            <h1 class="hero-container-title">
              <span class="container-badge mono">{{ item.container_number || 'UNCONFIRMED / REVIEW NEEDED' }}</span>
              <span class="status-pill" [class.confirmed]="item.status === 'CONFIRMED'" [class.review]="item.status !== 'CONFIRMED'">
                {{ item.status.replaceAll('_', ' ') }}
              </span>
              <span class="movement-pill">{{ item.event_type }}</span>
              @if (item.container_size) {
                <span class="size-pill feet">📏 {{ item.container_size }}{{ item.size_code ? ' (' + item.size_code + ')' : '' }}</span>
              }
              @if (hasFeetDetection(item)) {
                <span class="feet-pill">🦶 FEET DETECTED</span>
              }
            </h1>

            <div class="hero-meta-row">
              <span class="meta-item"><strong>Lane:</strong> {{ item.gate_id }}</span>
              <span class="meta-dot">·</span>
              <span class="meta-item"><strong>Time:</strong> {{ item.timestamp | date:'medium' }}</span>
              <span class="meta-dot">·</span>
              <span class="meta-item"><strong>Snapshots:</strong> {{ item.snapshots.length }} viewpoints</span>
              <span class="meta-dot">·</span>
              <span class="meta-item mono id-meta"><strong>ID:</strong> {{ item.id }}</span>
            </div>
          </div>
        </div>

        @if (showJson()) {
          <div class="json-viewer-panel">
            <div class="json-header">
              <div>
                <strong>event.json</strong>
                <small class="json-path">snapshots/{{ item.id }}/event.json</small>
              </div>
              <div class="json-actions">
                <button class="copy-btn" (click)="copyJson(item)">
                  {{ copied() ? '✓ Copied!' : 'Copy JSON' }}
                </button>
              </div>
            </div>
            <pre class="json-body"><code>{{ formatJson(item) }}</code></pre>
          </div>
        }

        <div class="evidence-columns">
          <div class="evidence-left">
            <div class="panel-card">
              <div class="panel-card-title">
                <h3>OCR Recognition Evidence</h3>
                <span class="badge">{{ item.ocr_results.length }} readings</span>
              </div>
              @if (item.ocr_results.length) {
                <div class="ocr-cards-list">
                  @for (ocr of item.ocr_results; track ocr.id) {
                    <div class="ocr-card" [class.ocr-valid]="ocr.validation_status === 'VALID' || ocr.validation_status === 'VALID_SIZE_CODE'">
                      <div class="ocr-card-header">
                        <span class="mono ocr-text">{{ ocr.normalized_text || 'No text' }}</span>
                        <span class="ocr-badge" [class.good]="ocr.validation_status === 'VALID' || ocr.validation_status === 'VALID_SIZE_CODE'">
                          {{ ocr.validation_status === 'VALID_SIZE_CODE' ? 'VALID SIZE CODE' : ocr.validation_status }}
                        </span>
                      </div>
                      <div class="ocr-card-details">
                        <div class="detail-row">
                          <span>Confidence:</span>
                          <strong>{{ ocr.confidence | percent:'1.0-1' }}</strong>
                        </div>
                        <div class="detail-row">
                          <span>Raw reading:</span>
                          <span class="mono">{{ ocr.raw_text || '(empty)' }}</span>
                        </div>
                      </div>
                    </div>
                  }
                </div>
              } @else {
                <p class="empty-hint">No OCR readings captured for this event.</p>
              }
            </div>

            <div class="panel-card" style="margin-top: 20px;">
              <div class="panel-card-title">
                <h3>Visual Detection Tracks</h3>
                <span class="badge">{{ item.detections.length }} objects</span>
              </div>
              @if (item.detections.length) {
                <div class="detection-list">
                  @for (d of item.detections; track d.id) {
                    <div class="detection-item" [class.feet-detection]="d.class_name.toLowerCase() === 'feet'">
                      <div class="detection-top">
                        <strong [class.feet-strong]="d.class_name.toLowerCase() === 'feet'">
                          {{ d.class_name.toUpperCase() }}
                          @if (d.class_name.toLowerCase() === 'feet') {
                            <span class="feet-badge-mini">TWISTLOCK / SIZE</span>
                          }
                        </strong>
                        <span class="mono">Camera: {{ d.camera_id }}</span>
                      </div>
                      <div class="detection-sub">
                        <span>Track #{{ d.track_id }}</span>
                        <span>Confidence: {{ d.confidence | percent:'1.0-1' }}</span>
                      </div>
                    </div>
                  }
                </div>
              } @else {
                <p class="empty-hint">No detection records found.</p>
              }
            </div>
          </div>

          <div class="evidence-right">
            <div class="panel-card">
              <div class="panel-card-title">
                <h3>Captured Multi-Camera Snapshots</h3>
                <span class="badge">{{ item.snapshots.length }} images</span>
              </div>
              <div class="snapshots-grid">
                @for (s of item.snapshots; track s.id) {
                  <figure class="snapshot-card">
                    <div class="snapshot-img-wrap">
                      <img [src]="'/api/snapshots/' + s.id" [alt]="'Snapshot from ' + s.camera_id" loading="lazy">
                      <span class="camera-tag">{{ s.camera_id }}</span>
                    </div>
                    <figcaption>
                      <span class="snap-time">{{ s.timestamp | date:'HH:mm:ss · dd MMM yyyy' }}</span>
                      <a [href]="'/api/snapshots/' + s.id" target="_blank" class="view-raw-link">Full resolution ↗</a>
                    </figcaption>
                  </figure>
                } @empty {
                  <p class="empty-hint">No camera snapshots recorded for this event.</p>
                }
              </div>
            </div>
          </div>
        </div>
      </div>
    }
  `
})
export class EventDetailComponent {
  event = input<EventDetail | null>(null);
  back = output<void>();
  showJson = signal(false);
  copied = signal(false);

  toggleJson() {
    this.showJson.update(v => !v);
  }

  hasFeetDetection(item: EventDetail | null): boolean {
    if (!item?.detections) return false;
    return item.detections.some(d => d.class_name?.toLowerCase() === 'feet');
  }

  formatJson(data: EventDetail): string {
    return JSON.stringify(data, null, 2);
  }

  copyJson(data: EventDetail) {
    const text = JSON.stringify(data, null, 2);
    navigator.clipboard.writeText(text).then(() => {
      this.copied.set(true);
      setTimeout(() => this.copied.set(false), 2000);
    });
  }
}

