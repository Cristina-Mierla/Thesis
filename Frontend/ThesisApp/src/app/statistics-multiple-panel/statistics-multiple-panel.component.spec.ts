import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StatisticsMultiplePanelComponent } from './statistics-multiple-panel.component';

describe('StatisticsMultiplePanelComponent', () => {
  let component: StatisticsMultiplePanelComponent;
  let fixture: ComponentFixture<StatisticsMultiplePanelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ StatisticsMultiplePanelComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(StatisticsMultiplePanelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
