import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { PatientFormComponent } from './patient-form/patient-form.component';
import { PatientListComponent } from './patient-list/patient-list.component';

const routes: Routes = [
  { path: '', component: PatientListComponent },
  { path: 'asses-new-patient', component: PatientFormComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
