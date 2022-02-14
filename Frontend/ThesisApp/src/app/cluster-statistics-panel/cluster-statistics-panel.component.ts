import { Component, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { PatientsService } from '../services/patients.service';

@Component({
  selector: 'app-cluster-statistics-panel',
  templateUrl: './cluster-statistics-panel.component.html',
  styleUrls: ['./cluster-statistics-panel.component.scss']
})
export class ClusterStatisticsPanelComponent implements OnInit {

  image?: Blob;
  imageUrl?: Object;
  showImage: boolean = false;
  age!: string;
  gender: number = 1;

  constructor(private patientsService: PatientsService, private sanitizer: DomSanitizer) { }

  ngOnInit(): void {
  }

  async getStatistic() {
    this.image = await this.patientsService.getClusterStatistics('/stat3', Number(this.age), this.gender)

    let unsafeImageUrl = URL.createObjectURL(this.image);
    this.imageUrl = this.sanitizer.bypassSecurityTrustUrl(unsafeImageUrl);
    this.showImage = true;
  }

  setMaleGender() {this.gender = 1;}

  setFemaleGender() {this.gender = 0;}
}
