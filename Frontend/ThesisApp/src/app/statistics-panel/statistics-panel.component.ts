import { Component, Input, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { Router } from '@angular/router';
import { PatientsService } from '../services/patients.service';

@Component({
  selector: 'app-statistics-panel',
  templateUrl: './statistics-panel.component.html',
  styleUrls: ['./statistics-panel.component.scss']
})
export class StatisticsPanelComponent implements OnInit {

  title: string = "";
  currentUrl: string = this.route.url;
  image?: Blob;
  imageUrl?: Object;

  constructor(private route: Router, private patientsService: PatientsService, private sanitizer: DomSanitizer) { }

  ngOnInit(): void {
    if (this.currentUrl === "/stat1"){
      this.title = "Outcome over age groups";
    }
    if (this.currentUrl === "/stat2"){
      this.title = "Distribution of age";
    }
    this.getStatistic();
  }

  async getStatistic() {
    this.image = await this.patientsService.getStatistics(this.currentUrl);

    let unsafeImageUrl = URL.createObjectURL(this.image);
    this.imageUrl = this.sanitizer.bypassSecurityTrustUrl(unsafeImageUrl);
  }

}
