import { Component, input } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { Health, Metrics } from './gate-api.service';

@Component({
  selector: 'gate-system-health', standalone: true, imports: [DecimalPipe],
  template: `<aside class="system-panel" aria-label="System health and GPU telemetry">
    <div class="system-heading"><h2>System health</h2><span class="status-dot" [class.online]="!stale() && health()?.status === 'READY'"></span></div>
    <p class="system-caption">{{ stale() ? 'Telemetry connection lost' : 'Edge computer · current readings' }}</p>
    <div class="health-line"><span>Database</span><strong [class.error-text]="health()?.database !== 'READY'">{{ health()?.database || 'CHECKING' }}</strong></div>
    <div class="health-line"><span>Detector</span><strong>{{ health()?.models?.detector || '—' }}</strong></div>
    <div class="health-line"><span>OCR engine</span><strong>{{ health()?.models?.ocr || '—' }}</strong></div>
    <div class="health-line"><span>OCR queue</span><strong>{{ metrics()?.ocr_queue_depth ?? 0 }} jobs</strong></div>
    <div class="resource"><div><span>CPU utilization</span><strong>{{ stale() || metrics()?.cpu_percent == null ? '—' : (metrics()?.cpu_percent | number:'1.0-0') + '%' }}</strong></div><progress [value]="stale() ? 0 : metrics()?.cpu_percent || 0" max="100" aria-label="CPU utilization"></progress></div>
    <div class="resource"><div><span>System memory</span><strong>{{ stale() || metrics()?.memory_percent == null ? '—' : (metrics()?.memory_percent | number:'1.0-0') + '%' }}</strong></div><progress [value]="stale() ? 0 : metrics()?.memory_percent || 0" max="100" aria-label="System memory utilization"></progress></div>
    <section class="gpu-panel" aria-label="GPU power"><div class="gpu-heading"><span class="chip-icon">GPU</span><small>ACCELERATOR</small></div><h3>{{ metrics()?.gpu_name || 'GPU telemetry unavailable' }}</h3>
      <small>POWER DRAW</small><div class="gpu-power">{{ stale() || metrics()?.gpu_power_watts == null ? '—' : (metrics()?.gpu_power_watts | number:'1.1-1') }} <span>W</span></div>
      @if (metrics()?.gpu_power_limit_watts != null && !stale()) {<small>Power limit {{ metrics()?.gpu_power_limit_watts | number:'1.0-0' }} W</small>}
      <div class="gpu-values"><div><small>UTILIZATION</small><strong>{{ stale() || metrics()?.gpu_utilization_percent == null ? '—' : (metrics()?.gpu_utilization_percent | number:'1.0-0') + '%' }}</strong></div><div><small>TEMPERATURE</small><strong>{{ stale() || metrics()?.gpu_temperature_c == null ? '—' : (metrics()?.gpu_temperature_c | number:'1.0-0') + ' °C' }}</strong></div></div>
      <div class="gpu-memory"><small>GPU MEMORY</small><span>{{ stale() || metrics()?.gpu_memory_used_mb == null ? 'Unavailable' : (metrics()?.gpu_memory_used_mb | number:'1.0-0') + ' / ' + (metrics()?.gpu_memory_total_mb | number:'1.0-0') + ' MB' }}</span></div>
      @if (metrics()?.gpu_power_watts == null) {<p class="telemetry-note">Power readings are not exposed by this device or driver.</p>}
    </section>
    <div class="context-block"><small>CONNECTED SITE</small><strong>Not configured</strong><span>Site setup deferred</span></div>
    <div class="context-block"><small>ADMIN SESSION</small><strong>Not signed in</strong><span>Local testing · login setup deferred</span></div>
  </aside>`
})
export class SystemHealthComponent {
  health = input<Health | null>(null);
  metrics = input<Metrics | null>(null);
  stale = input(false);
}
