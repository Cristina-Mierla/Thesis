import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { PatientFormComponent } from './patient-form/patient-form.component';
import { PatientListComponent } from './patient-list/patient-list.component';
import { PatientDetailsComponent } from './patient-details/patient-details.component';
import { query } from '@angular/animations';
import { StatisticsPanelComponent } from './statistics-panel/statistics-panel.component';
import { ClusterStatisticsPanelComponent } from './cluster-statistics-panel/cluster-statistics-panel.component';
import { StatisticsMultiplePanelComponent } from './statistics-multiple-panel/statistics-multiple-panel.component';

const routes: Routes = [
  { path: '', component: PatientListComponent },
  { path: 'asses-new-patient', component: PatientFormComponent },
  { path: 'patient/:id', component: PatientDetailsComponent},
  { path: `patient`, component: PatientDetailsComponent},
  { path: 'stat1', component: StatisticsPanelComponent},
  { path: 'stat2', component: StatisticsPanelComponent},
  { path: 'stat3', component: ClusterStatisticsPanelComponent},
  { path: 'stat4', component: StatisticsMultiplePanelComponent}

];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
