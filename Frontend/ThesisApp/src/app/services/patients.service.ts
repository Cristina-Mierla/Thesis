import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/environment';
import { BehaviorSubject, firstValueFrom, Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
import { PatientItem } from '../models/patientItem';
import { Patient } from '../models/patient';

@Injectable({
  providedIn: 'root'
})
export class PatientsService {
  //patientList!: Observable<string[]>;
  //private patientId = new BehaviorSubject(0);
  //currentMessage = this.getPatientById(0);
  private patientData!: Promise<Patient>;
  private patientIdData!: number;

  constructor(private http: HttpClient) { }

  public readonly patientsUrl = `${environment.apiUrl}`;

  public async getPatients(): Promise<PatientItem[]>{
    // this.http.get<any>(`${this.patientsUrl}patients`).subscribe(data => {
    //   this.patientList = data;
    // })
    // return this.patientList;
    return await firstValueFrom(this.http.get<PatientItem[]>(`${this.patientsUrl}patients`));
  }

  public async getPatientById(id : number): Promise<Patient> {
    return await firstValueFrom(this.http.get<Patient>(`${this.patientsUrl}patientId?patient_id=${id}`));
  }

  public async makePrediction(patient: Patient): Promise<string> {
    return await firstValueFrom(this.http.post<string>(`${this.patientsUrl}prediction`, patient));
  }

  public async getStatistics(url: string): Promise<Blob> {
    return await firstValueFrom(this.http.get(`${this.patientsUrl}${url}`, { responseType: 'blob' }));
  }

  public async getClusterStatistics(url: string, age: number, gender: number): Promise<Blob> {
    return await firstValueFrom(this.http.get(`${this.patientsUrl}${url}?age=${age}&gender=${gender}`, { responseType: 'blob' }));
  }

  public setPatientIdData(data: number){
    this.patientIdData = data;
  }

  public getPatientIdData(): number{
    return this.patientIdData;
  }

  public setPatientData(data: Promise<Patient>){
    this.patientData = data;
  }

  public getPatientData(): Promise<Patient>{
    return this.patientData;
  }
}
