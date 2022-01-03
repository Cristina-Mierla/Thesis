import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Patient } from '../models/patient';
import { PatientsService } from '../services/patients.service';

@Component({
  selector: 'app-patient-details',
  templateUrl: './patient-details.component.html',
  styleUrls: ['./patient-details.component.scss']
})
export class PatientDetailsComponent implements OnInit {

  patient!: Patient;
  public id: number = 0;

  constructor(private patientsService: PatientsService, private route: ActivatedRoute) { }

  ngOnInit(): void {
    this.getPatientId();
    this.id = Number(this.route.snapshot.paramMap.get('id'));
    this.loadPatientsById(this.id);
  }

  getPatientId(){
    this.id = this.patientsService.getPatientIdData();
  }

  async loadPatientsById(id: number){
    this.patient = await this.patientsService.getPatientById(id);
  }
}
