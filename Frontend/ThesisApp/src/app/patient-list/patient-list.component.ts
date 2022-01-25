import { AfterViewInit, Component, ViewChild, OnInit } from '@angular/core';
import { Observable } from 'rxjs';
import { GlobalConstants } from '../constants/constants';
import { ReleaseState } from '../enums/releaseState';
import { ReleaseStateColorMap } from '../enums/releaseStateColorMap';
import { PatientItem } from '../models/patientItem';
import { PatientFormComponent } from '../patient-form/patient-form.component';
import { PatientsService } from '../services/patients.service';
import { MatPaginator } from '@angular/material/paginator';
import { MatTableDataSource } from '@angular/material/table';


@Component({
  selector: 'app-patient-list',
  templateUrl: './patient-list.component.html',
  styleUrls: ['./patient-list.component.scss']
})
export class PatientListComponent implements OnInit {

  colorMap = ReleaseStateColorMap;
  release = ReleaseState;
  patientList: PatientItem[] = [];
  p: number = 1;

  constructor(private patientsService: PatientsService) {}

  displayedColumns: string[] = ['details', 'outcome', 'view'];
  dataSource = new MatTableDataSource<PatientItem>(this.patientList);
  @ViewChild(MatPaginator) paginator!: MatPaginator;

  ngOnInit(): void {
    this.loadPatients();
  }

  ngAfterViewInit(): void {
    this.dataSource.paginator = this.paginator;
  }

  async loadPatients() {
    this.patientList = await this.patientsService.getPatients();
    this.dataSource = new MatTableDataSource<PatientItem>(this.patientList);
    this.dataSource.paginator = this.paginator;
  }

  async loadPatientId(id: number) {
    this.patientsService.setPatientIdData(id);
  }

 }
