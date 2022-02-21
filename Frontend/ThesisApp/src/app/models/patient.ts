export interface Patient{
  Id: number,
  Age: number,
  Gender: number,
  Hosp: number,
  Icu: number,
  Diag: string,
  Comb: string[],
  Med: string[],
  Anlz: string[],
  Release: number
}
