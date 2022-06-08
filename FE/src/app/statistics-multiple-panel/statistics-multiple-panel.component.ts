import { Component, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { PatientsService } from '../services/patients.service';

@Component({
  selector: 'app-statistics-multiple-panel',
  templateUrl: './statistics-multiple-panel.component.html',
  styleUrls: ['./statistics-multiple-panel.component.scss']
})
export class StatisticsMultiplePanelComponent implements OnInit {

  image!: Blob[];
  imageUrl?: Object;
  imageUrl1?: Object;
  imageUrl2?: Object;
  loaded = false;

  constructor(private patientsService: PatientsService, private sanitizer: DomSanitizer) { }

  ngOnInit(): void {
    this.getStatisticMultiple();
  }

  async getStatisticMultiple() {
    await this.patientsService.getStatisticsMultiple('/stat4')
      .then((x) => {this.image = x; this.loaded = true;});

    let unsafeImageUrl1 = URL.createObjectURL(this.image[0]);
    let unsafeImageUrl2 = URL.createObjectURL(this.image[1]);
    this.imageUrl1 = this.sanitizer.bypassSecurityTrustUrl(unsafeImageUrl1);
    this.imageUrl2 = this.sanitizer.bypassSecurityTrustUrl(unsafeImageUrl2);
  }

}
