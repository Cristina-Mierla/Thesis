import numpy as np
import pandas as pd
import csv

class Service:
    def __init__(self, file):
        try:
            dataset = pd.read_csv(file)
            self.df = pd.DataFrame(dataset)
            self.df.stare_externare.replace(
                ("Vindecat", "Ameliorat", "Stationar", "AGRAVAT                                           ", "Decedat"),
                (0, 1, 2, 3, 4), inplace=True)
        except IOError:
            print("The file does not exist")

    def getPatientHeadList(self):
        array = []
        for line in self.df.values:
            record = pd.DataFrame(line)
            pid = record.iloc[1, 0]
            releaseState = str(record.iloc[18, 0])
            gender = record.iloc[7, 0]
            age = record.iloc[8, 0]
            name = "Age:" + str(age) + " Gender:" + str(gender)
            patient = {"Id": pid, "Details": name, "Outcome": releaseState}
            array.append(patient)
        return array

    def getPatientById(self, id):
        patient = None
        indx = 0
        for line in self.df.values:
            record = pd.DataFrame(line)
            if record.iloc[1, 0] == id:
                patient = record
                break
            indx += indx
        zspit = patient.iloc[5, 0]
        zicu = patient.iloc[6, 0]
        gender = patient.iloc[7, 0]
        age = patient.iloc[8, 0]
        comorb = patient.iloc[9, 0]
        med = patient.iloc[10, 0]
        analz = patient.iloc[11, 0]
        release = patient.iloc[18, 0]

        print(patient)


if __name__ == '__main__':
    s = Service("csv_dataset.csv")
    s.getPatientHeadList()
    s.getPatientById(19904)
    # d.pltReleaseState()