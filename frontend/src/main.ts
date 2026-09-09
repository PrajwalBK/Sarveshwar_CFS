import { bootstrapApplication } from '@angular/platform-browser';
import { provideHttpClient } from '@angular/common/http';
import { provideZonelessChangeDetection } from '@angular/core';
import { AppComponent } from './app/app.component';

bootstrapApplication(AppComponent, {providers: [provideHttpClient(), provideZonelessChangeDetection()]})
  .catch(() => { document.body.textContent = 'Gate UI could not start. Reload or contact the system operator.'; });
