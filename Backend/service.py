from imports import *

from Backend.dataAnalysis import DataAnalysis
from Backend.dataProcessing import DataProcessing
from Backend.predictionModel import PredictionModel


class Service:
    def __init__(self, file):
        print("Service")
        data = DataAnalysis("csv_dataset.csv")
        self.initialDf = data.getDataset()
        self.processData = DataProcessing()
        self.dataAnalysis = DataAnalysis("")

        self.df = self.processData.getDataset()
        self.comorb = self.processData.getReadComorbiditati()
        self.medicatie = self.processData.getMedicatie()
        self.analize = self.processData.getAnalize()

        self.dataAnalysis.setDataset(self.df)

        self.modelClass = PredictionModel(self.df)

    def getPatientHeadList(self):
        array = []
        print(self.df.columns)
        for line in self.df.values:
            record = pd.DataFrame(line)

            pid = record.iloc[1, 0]
            releaseState = record.iloc[10, 0]
            gender = record.iloc[6, 0]
            age = record.iloc[7, 0]
            # name = "Age:" + str(age) + " Gender:" + str(gender)
            patient = {"Id": pid, "Age": age, "Gender": gender, "Outcome": releaseState}
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
            zspit = patient.iloc[4, 0]
            zicu = patient.iloc[5, 0]
            gender = patient.iloc[6, 0]
            age = patient.iloc[7, 0]
            comorb = self.comorb[str(indx)]
            comorb = [x for x in comorb if x != '']
            med = self.medicatie[str(indx)]
            med = [x for x in med if x != '']
            analz = self.analize[str(indx)]
            analz = [x for x in analz if x != '']
            release = patient.iloc[10, 0]

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

    def makePrediction(self, prediction_data):
        # self.model = pickle.load(open('model.pkl', 'rb'))
        print(prediction_data)
        prediction_set = self.processData.prediction(prediction_data)
        prediction_result = self.modelClass.predict(prediction_set)

        return prediction_result

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

        image = self.dataAnalysis.categorizeData(filename, age, gender)
        return image

    def getStatistics4(self):
        dt_string = datetime.now().strftime("_%Y-%m-%d_%H-%M")
        filename = "statistics4\\computedClusteres_" + dt_string

        image1, image2 = self.modelClass.clusteringDataWithPCA(filename)
        return image1, image2


if __name__ == '__main__':
    s = Service("csv_processedDataset.csv")
    s.getPatientHeadList()
    s.getPatientById(19904)
    s.getStatistics3(53, 1)
    # d.pltReleaseState()
