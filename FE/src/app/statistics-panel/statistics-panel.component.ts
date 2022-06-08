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
  caption: string = "";
  currentUrl: string = this.route.url;
  image!: Blob;
  imageUrl?: Object;
  loaded = false;

  constructor(private route: Router, private patientsService: PatientsService, private sanitizer: DomSanitizer) { }

  ngOnInit(): void {
    if (this.currentUrl === "/stat1"){
      this.title = "Outcome over age groups";
      this.caption = "Graph describing how many people were hospitalized in different age groups.";
    }
    if (this.currentUrl === "/stat2"){
      this.title = "Distribution of age based on outcome";
      this.caption = "Distribution of the days spent in hospitalization over all ages present in the dataset, with a different hue for every release state.";
    }
    this.getStatistic();
  }

  async getStatistic() {
    await this.patientsService.getStatistics(this.currentUrl)
      .then((x) => {this.image = x; this.loaded = true;})

    let unsafeImageUrl = URL.createObjectURL(this.image);
    this.imageUrl = this.sanitizer.bypassSecurityTrustUrl(unsafeImageUrl);
  }

}
