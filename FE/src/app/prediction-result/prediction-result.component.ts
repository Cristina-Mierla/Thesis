import {Component, Inject} from '@angular/core';
import { MatDialog, MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { PatientFormComponent } from '../patient-form/patient-form.component';
import { Prediction } from '../models/predictionResult';

@Component({
  selector: 'app-prediction-result',
  templateUrl: './prediction-result.component.html',
  styleUrls: ['./prediction-result.component.scss']
})
export class PredictionResultComponent {

  constructor(public dialogRef: MatDialogRef<PatientFormComponent>,
    @Inject(MAT_DIALOG_DATA) public data: string,) { }

  onNoClick(): void {
    this.dialogRef.close();
  }
}
