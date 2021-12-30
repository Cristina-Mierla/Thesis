import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { NavbarComponent } from './navbar/navbar.component';
import { PatientListComponent } from './patient-list/patient-list.component';
import { NgxPaginationModule } from 'ngx-pagination';
import { PatientFormComponent } from './patient-form/patient-form.component';
@NgModule({
  declarations: [
    AppComponent,
    NavbarComponent,
    PatientListComponent,
    PatientFormComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    AppRoutingModule,
    NgbModule,
    NgxPaginationModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
