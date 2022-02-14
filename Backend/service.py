import numpy as np
import pandas as pd
import csv
from datetime import date, datetime

from Backend.dataAnalysis import DataAnalysis
from Backend.dataProcessing import DataProcessing


class Service:
    def __init__(self, file):
        try:
            dataset = pd.read_csv(file)
            self.df = pd.DataFrame(dataset)
            # self.df.stare_externare.replace(
            #     ("Vindecat", "Ameliorat", "Stationar", "AGRAVAT                                           ", "Decedat"),
            #     (0, 1, 2, 3, 4), inplace=True)

            filename_comrb = "csv_comorbiditati.csv"
            filename_analize = "csv_analize.csv"
            filename_medicatie = "csv_medicatie.csv"

            with open(filename_comrb, mode='r') as infile:
                print("Dictionary saved from file " + filename_comrb)
                reader = csv.reader(infile)
                self.comorb = {rows[0]: rows[1:] for rows in reader}

            with open(filename_analize, mode='r') as infile:
                print("Dictionary saved from file " + filename_analize)
                reader = csv.reader(infile)
                self.analize = {rows[0]: rows[1:] for rows in reader}

            with open(filename_medicatie, mode='r') as infile:
                print("Dictionary saved from file " + filename_medicatie)
                reader = csv.reader(infile)
                self.medicatie = {rows[0]: rows[1:] for rows in reader}
        except IOError:
            print("The file does not exist")

        # self.processData = DataProcessing()
        self.dataAnalysis = DataAnalysis("")
        self.dataAnalysis.setDataset(self.df)

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
        result = None
        patient = None
        ok = False
        indx = 0
        for line in self.df.values:
            record = pd.DataFrame(line)
            if record.iloc[1, 0] == id:
                patient = record
                ok = True
                break
            indx += indx
        if ok:
            pid = patient.iloc[1, 0]
            zspit = patient.iloc[5, 0]
            zicu = patient.iloc[6, 0]
            gender = patient.iloc[7, 0]
            age = patient.iloc[8, 0]
            comorb = self.comorb[str(indx)]
            comorb = [x for x in comorb if x != '']
            med = self.medicatie[str(indx)]
            med = [x for x in med if x != '']
            analz = self.analize[str(indx)]
            analz = [x for x in analz if x != '']
            release = patient.iloc[18, 0]

            result = {
                'Id': pid,
                'Age': age,
                'Gender': gender,
                'Hosp': zspit,
                'Icu': zicu,
                'Comb': comorb,
                'Med': med,
                'Anlz': analz,
                'Release': release
            }

        return result

    def getStatistics1(self):
        dt_string = datetime.now().strftime("_%Y-%m-%d_%H-%M")
        filename = "statistics1\\ageGroup_" + dt_string

        image = self.dataAnalysis.groupAge(filename)
        return image

    def getStatistics2(self):
        dt_string = datetime.now().strftime("_%Y-%m-%d_%H-%M")
        filename = "statistics2\\ageDistribution_" + dt_string

        image = self.dataAnalysis.pltReleaseState(filename)
        return image

    def getStatistics3(self, age, gender):
        dt_string = datetime.now().strftime("_%Y-%m-%d_%H-%M")
        filename = "statistics3\\clusterData_" + dt_string

        image = self.dataAnalysis.clusteringData(filename, age, gender)
        return image


if __name__ == '__main__':
    s = Service("csv_processedDataset.csv")
    s.getPatientHeadList()
    s.getPatientById(19904)
    s.getStatistics3(53, 1)
    # d.pltReleaseState()