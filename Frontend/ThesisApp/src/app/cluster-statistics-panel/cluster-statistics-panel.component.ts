import { Component, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { PatientsService } from '../services/patients.service';
import {
  MatSnackBar,
  MatSnackBarHorizontalPosition,
  MatSnackBarVerticalPosition,
} from '@angular/material/snack-bar';

@Component({
  selector: 'app-cluster-statistics-panel',
  templateUrl: './cluster-statistics-panel.component.html',
  styleUrls: ['./cluster-statistics-panel.component.scss']
})
export class ClusterStatisticsPanelComponent implements OnInit {

  image!: Blob;
  imageUrl?: Object;
  showImage: boolean = false;
  age!: string;
  gender: number = 1;
  loaded = true;

  constructor(private patientsService: PatientsService,
              private sanitizer: DomSanitizer,
              private _snackBar: MatSnackBar) { }

  ngOnInit(): void {
  }

  async getStatistic() {
    if (!(this.age && Number(this.age))) {
      let snackBarRef = this._snackBar.open('Please input a valid age', 'Ok', {
             horizontalPosition: 'center', verticalPosition: 'bottom',
           });
    } else {
      this.loaded = false;
      await this.patientsService.getClusterStatistics('/stat3', Number(this.age), this.gender)
        .then((x) => {this.image = x; this.loaded = true;})

      let unsafeImageUrl = URL.createObjectURL(this.image);
      this.imageUrl = this.sanitizer.bypassSecurityTrustUrl(unsafeImageUrl);
      this.showImage = true;
    }
  }

  setMaleGender() {this.gender = 1;}

  setFemaleGender() {this.gender = 0;}
}
