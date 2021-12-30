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
            name = str(pid) + " age:" + str(age) + " gender:" + str(gender)
            patient = {"Name": name, "Outcome": releaseState}
            array.append(patient)
        return array


if __name__ == '__main__':
    s = Service("csv_dataset.csv")
    s.getPatientHeadList()
    # d.pltReleaseState()