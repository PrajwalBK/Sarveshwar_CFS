import { Component, input, output } from '@angular/core';
import { DatePipe } from '@angular/common';
import { GateEvent } from './gate-api.service';

@Component({
  selector: 'gate-event-table', standalone: true, imports: [DatePipe],
  template: `<div class="table-scroll"><table><thead><tr><th>Time</th><th>Container number</th><th>Lane</th><th>Movement</th><th>Processing status</th><th>Evidence & Actions</th></tr></thead>
  <tbody>@for (event of events(); track event.id) {<tr>
    <td>{{ event.timestamp | date:'dd MMM, HH:mm:ss' }}</td>
    <td class="mono font-bold">{{ event.container_number || 'Unread / unconfirmed' }}</td>
    <td>{{ event.gate_id }}</td>
    <td><span class="badge movement-badge">{{ event.event_type }}</span></td>
    <td><span class="badge" [class.good]="event.status === 'CONFIRMED'" [class.warn]="event.status === 'NEEDS_REVIEW'">{{ event.status.replaceAll('_', ' ') }}</span></td>
    <td>
      <div class="action-cell">
        <button class="view-event-btn" (click)="selected.emit(event.id)">View event →</button>
        <a class="json-badge-link" [href]="'/api/gate-events/' + event.id + '/json'" [download]="'event_' + (event.container_number || event.id) + '.json'" title="Download event.json">
          event.json
        </a>
      </div>
    </td>
  </tr>} @empty {<tr><td colspan="6" class="empty">No events match this view. Events appear after sufficient detection and OCR evidence.</td></tr>}</tbody></table></div>`
})
export class EventTableComponent {
  events = input<GateEvent[]>([]);
  selected = output<string>();
}
