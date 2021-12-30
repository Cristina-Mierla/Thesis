import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/environment';
import { firstValueFrom, Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
import { PatientItem } from '../models/patientItem';

@Injectable({
  providedIn: 'root'
})
export class PatientsService {
  //patientList!: Observable<string[]>;

  constructor(private http: HttpClient) { }

  public readonly patientsUrl = `${environment.apiUrl}`;

  public async getPatients(): Promise<PatientItem[]>{
    // this.http.get<any>(`${this.patientsUrl}patients`).subscribe(data => {
    //   this.patientList = data;
    // })
    // return this.patientList;
    return await firstValueFrom(this.http.get<PatientItem[]>(`${this.patientsUrl}patients`));
  }
}
