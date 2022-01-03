import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { PatientFormComponent } from './patient-form/patient-form.component';
import { PatientListComponent } from './patient-list/patient-list.component';
import { PatientDetailsComponent } from './patient-details/patient-details.component';
import { query } from '@angular/animations';

const routes: Routes = [
  { path: '', component: PatientListComponent },
  { path: 'asses-new-patient', component: PatientFormComponent },
  { path: 'patient/:id', component: PatientDetailsComponent},
  { path: `patient`, component: PatientDetailsComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
