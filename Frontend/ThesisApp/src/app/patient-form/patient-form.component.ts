import { Component, OnInit } from '@angular/core';
import { Patient } from '../models/patient';
import { PatientsService } from '../services/patients.service';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-patient-form',
  templateUrl: './patient-form.component.html',
  styleUrls: ['./patient-form.component.scss']
})
export class PatientFormComponent implements OnInit {

  constructor(private patientsService: PatientsService, private route: ActivatedRoute) { }

  ngOnInit(): void {
  }

}
