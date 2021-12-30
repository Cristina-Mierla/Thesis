import { Component, OnInit } from '@angular/core';
import { Observable } from 'rxjs';
import { GlobalConstants } from '../constants/constants';
import { ReleaseState } from '../enums/releaseState';
import { ReleaseStateColorMap } from '../enums/releaseStateColorMap';
import { PatientItem } from '../models/patientItem';
import { PatientsService } from '../services/patients.service';


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

  ngOnInit(): void {
    this.loadPatients();
  }

  async loadPatients(){
    this.patientList = await this.patientsService.getPatients();
  }

 }
