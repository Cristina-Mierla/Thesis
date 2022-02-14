import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ClusterStatisticsPanelComponent } from './cluster-statistics-panel.component';

describe('ClusterStatisticsPanelComponent', () => {
  let component: ClusterStatisticsPanelComponent;
  let fixture: ComponentFixture<ClusterStatisticsPanelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ClusterStatisticsPanelComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ClusterStatisticsPanelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
