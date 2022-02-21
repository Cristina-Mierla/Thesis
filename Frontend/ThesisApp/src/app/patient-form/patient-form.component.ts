import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { Patient } from '../models/patient';
import { PatientsService } from '../services/patients.service';
import { ActivatedRoute } from '@angular/router';
import { COMMA, ENTER } from '@angular/cdk/keycodes';
import { Observable } from 'rxjs';
import { FormControl } from '@angular/forms';
import { map, startWith } from 'rxjs/operators';
import { MatAutocompleteSelectedEvent } from '@angular/material/autocomplete';
import { MatChipInputEvent } from '@angular/material/chips';
import { comorbiditiList } from '../constants/constants';
import { MatSnackBar } from '@angular/material/snack-bar';
@Component({
  selector: 'app-patient-form',
  templateUrl: './patient-form.component.html',
  styleUrls: ['./patient-form.component.scss']
})
export class PatientFormComponent implements OnInit {

  separatorKeysCodes: number[] = [ENTER, COMMA];
  comorbiditiesCtrl = new FormControl();
  diagnosisCtrl = new FormControl();
  filteredComorbidities: Observable<string[]>;
  //allComorbiditiesObservable: Observable<string[]>;
  allComorbidities: string[] = comorbiditiList;

  chosenComorbidities: string[] = [];
  initialDiagnosis?: string = undefined;
  age!: number;
  zspital!: number;
  zicu!: number;
  gender!: number;

  newPatient?: Patient;

  @ViewChild('comorbInput') comorbInput!: ElementRef<HTMLInputElement>;
  @ViewChild('diagnosInput') diagnosInput!: ElementRef<HTMLInputElement>;

  constructor(private patientsService: PatientsService,
    private route: ActivatedRoute,
    private _snackBar: MatSnackBar) {
    this.filteredComorbidities = this.comorbiditiesCtrl.valueChanges.pipe(
      startWith(null),
      map((comorb: string | null) => (comorb ? this._filter(comorb) : this.allComorbidities.slice())),
    );

   }

  ngOnInit(): void {
  }

  addComorbidity(event: MatChipInputEvent): void {
    const value = (event.value || '').trim();

    // Add new comorbiditie
    if (value && this.chosenComorbidities.indexOf(value) == -1) {
      this.chosenComorbidities.push(value);
    }

    // Clear the input value
    event.chipInput!.clear();

    this.comorbiditiesCtrl.setValue(null);
  }

  addDiagnosis(event: MatChipInputEvent): void {
    const value = (event.value || '').trim();

    // Choose comorbidity
    if (value){
      this.initialDiagnosis = value;
    }

    // Clear the input value
    event.chipInput!.clear();

    this.diagnosisCtrl.setValue(null);
  }

  removeComorbidity(comorb: string): void {
    const index = this.chosenComorbidities.indexOf(comorb);

    if (index >= 0) {
      this.chosenComorbidities.splice(index, 1);
    }
  }

  removeDiagnosis(): void {
    this.initialDiagnosis = undefined;
  }

  selectedComorbidity(event: MatAutocompleteSelectedEvent): void {
    const value = (event.option.viewValue || '').trim();

    if (value && this.chosenComorbidities.indexOf(value) == -1) {
      this.chosenComorbidities.push(event.option.viewValue);
    }

    //this.fruits.push(event.option.viewValue);
    this.comorbInput.nativeElement.value = '';
    this.comorbiditiesCtrl.setValue(null);
  }

  selectedDiagnosis(event: MatAutocompleteSelectedEvent): void {
    const value = (event.option.viewValue || '').trim();

    if (value){
      this.initialDiagnosis = event.option.viewValue
    }

    this.diagnosInput.nativeElement.value = '';
    this.diagnosisCtrl.setValue(null);
  }

  private _filter(value: string): string[] {
    const filterValue = value.toLowerCase();

    return this.allComorbidities.filter(comorb => comorb.toLowerCase().includes(filterValue));
  }

  setMaleGender() {this.gender = 1;}

  setFemaleGender() {this.gender = 0;}

  submitData(): Patient | undefined {
    if (!(this.age && this.gender && this.zspital && this.zicu && this.initialDiagnosis && this.chosenComorbidities)){
      let snackBarRef = this._snackBar.open('Please input a valid age', 'Ok', {
        horizontalPosition: 'center', verticalPosition: 'bottom',
      });
    } else {
      this.newPatient = {
        Id: 0,
        Age: this.age,
        Gender: this.gender,
        Hosp: this.zspital,
        Icu: this.zicu,
        Diag: this.initialDiagnosis,
        Comb: [''],
        Med: [""],
        Anlz: [""],
        Release: 0
      };
    }

    return this.newPatient;
  }

}
