import numpy as np

from imports import *
from Backend.dataAnalysis import DataAnalysis

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class DataProcessing:
    def __init__(self):
        print("Processing the dataset")
        self.dataset = pd.DataFrame()
        # self.df = pd.DataFrame(self.dataset)
        self.df = pd.DataFrame()
        self.comorbiditati = None
        self.medicatie = None
        self.analize = None
        self.comorb = None

        filename_dataset = "csv_NewProcessedDataset.csv"
        filename_comorb = "csv_comorb_weight.csv"
        filename_comrb = "csv_comorbiditati.csv"
        filename_analize = "csv_analize.csv"
        filename_medicatie = "csv_medicatie.csv"

        self.file_comorb_opened = False
        self.file_comrb_opened = False
        self.file_analize_opened = False
        self.file_medicatie_opened = False
        self.file_dataset_opened = False

        try:
            with open(filename_comorb, mode='r') as infile:
                print("Dictionary saved from file " + filename_comorb + " in DataProcessing")
                reader = csv.reader(infile)
                self.comorbiditati = {rows[0]: rows[1] for rows in reader}
                self.file_comorb_opened = True

            with open(filename_comrb, mode='r') as infile:
                print("Dictionary saved from file " + filename_comrb + " in DataProcessing")
                reader = csv.reader(infile)
                self.comorb = {rows[0]: rows[1:] for rows in reader}
                self.file_comrb_opened = True

            with open(filename_analize, mode='r') as infile:
                print("Dictionary saved from file " + filename_analize + " in DataProcessing")
                reader = csv.reader(infile)
                self.analize = {rows[0]: rows[1:] for rows in reader}
                self.file_analize_opened = True

            with open(filename_medicatie, mode='r') as infile:
                print("Dictionary saved from file " + filename_medicatie + " in DataProcessing")
                reader = csv.reader(infile)
                self.medicatie = {rows[0]: rows[1:] for rows in reader}
                self.file_medicatie_opened = True

            with open(filename_dataset, mode='r') as infile:
                print("Dataset saved from file " + filename_dataset + " in DataProcessing")
                self.dataset = pd.read_csv(filename_dataset, parse_dates=True)
                self.df = pd.DataFrame(self.dataset)
                self.file_dataset_opened = True

        except IOError and FileNotFoundError:
            print("Files " + filename_dataset + " " + filename_comorb + " not found from request in DataProcessing")
            try:
                dataset = pd.read_csv("csv_dataset.csv")
                self.df = pd.DataFrame(dataset)
                print(self.df)

                self.replaceColumns()

                self.comorbidityCounts = None

                self.comorbiditati = self.changeComorbiditati()
                self.changeDiagnos()
                self.featureCross()
                self.medicatie = self.changeMedicatie()
                self.analize = self.changeAnalize()

                self.comorb = self.df["Comorbiditati"]
                self.exetern = self.df["stare_externare"]
                self.varsta = self.df["Varsta"]
                self.spitalizare = self.df["Zile_spitalizare"]
                self.diagnos_int = self.df["Diag_pr_int"]
                self.com_ext = self.df[["Comorbiditati", "stare_externare"]]

                self.df.to_csv(filename_dataset, header=True)
            except IOError and FileNotFoundError:
                print("The file cannot be find or read")

        self.df.rename(columns=lambda s: s.replace("*", "A").replace("(", "").replace(")", "").replace("%", "B")
                       .replace("#", "C").replace("/", "").replace(",", ''), inplace=True)

    def replaceColumns(self):
        """
        Removing redundant information.
        Replacing with Python NULL in the empty records.
        Making categorical data numerical.
        :return:
        """
        # print("\nDrop the columns that we are not going to use for the model")
        self.df.drop(["AN", "precod", "Data_Examinare_Radiologie", "Radiologie", "rezultat_radiologie", "Proceduri",
                      "Proceduri_Radiologie", "tip_externare", "unde_pleaca"], inplace=True, axis='columns')

        self.df.replace("NULL", np.NAN, inplace=True)
        self.df.replace("", np.NAN, inplace=True)
        self.df.replace("_", np.NAN, inplace=True)

        le = LabelEncoder()
        self.df["Sex"] = le.fit_transform(self.df["Sex"])
        self.df.stare_externare.replace(
            ("Vindecat", "Ameliorat", "Stationar", "AGRAVAT                                           ", "Decedat"),
            (0, 1, 2, 3, 4), inplace=True)
        self.df.forma_boala.replace(('1.USOARA', '2. MODERATA', '3.SEVERA', 'PERICLITAT TRANSFUZIONAL'),
                                    (1, 2, 3, np.NaN), inplace=True)

        self.df.forma_boala = self.df.forma_boala.fillna(self.df.forma_boala.median())

    def getDataset(self):
        return self.df

    def setDataset(self, dataset):
        self.df = pd.DataFrame(dataset, columns=self.df.columns)
        print(self.df)

    def getMedicatie(self):
        return self.medicatie

    def getAnalize(self):
        return self.analize

    def getComorbiditati(self):
        return self.comorbiditati

    def getReadComorbiditati(self):
        return self.comorb

    def compareTestResults(self):
        print(self.df)
        # leukocytes/Leucocite, UREA, Albuminum, Calcium, Creatinine/ProteinaCreactiva, AST,
        # Glucose/GLU, Neutrophils, Total protein

        leucocite = self.df["Leucocite"]
        urea = self.df["UREA"]
        proteinaCreactiva = self.df["ProteinaCreactiva"]
        glucose = self.df["GLU"]
        list = [leucocite, urea, proteinaCreactiva, glucose]
        a = 0
        i = j = 0

        fig, axs = plt.subplots(2, 2, constrained_layout=True)
        for x in list:
            axs[i, j].scatter(x, self.df['stare_externare'])
            axs[i, j].set_title(str(x.name))
            axs[i, j].set(ylabel="Release state")
            # plt.xlim([-2, 40])
            a += 1
            i = j = 1 if a == 3 else 0
            j = 1 if a == 1 else j
            i = 1 if a == 2 else i

        plt.show()

    def changeMedicatie(self):
        """
        One Hot Encoding for the "Medicatie" column.
        :return:
        """
        dm = {}
        indx = 0
        self.medicatie = dict()
        for record in self.df.Medicatie:
            med_list = str(record).split("|| ")
            self.medicatie[indx] = med_list
            for med in med_list:
                med = med.replace(" ", "")
                try:
                    self.df[med][indx] = 1
                except:
                    self.df[med] = np.zeros(self.df.shape[0], dtype=int)
                    self.df[med][indx] = 1
                    pd.to_numeric(self.df[med])
                dm[med] = 1
            indx += 1
        for key, value in dm.items():
            if self.df[key].sum() <= self.df.shape[0] * 0.2:
                self.df.drop([key], inplace=True, axis='columns')
        self.df.drop(['Medicatie'], inplace=True, axis='columns')
        if not self.file_medicatie_opened:
            csv_file = "csv_medicatie.csv"
            try:
                with open(csv_file, 'w') as f:
                    for key in self.medicatie.keys():
                        f.write("%s,%s\n" % (key, self.medicatie[key]))
            except IOError:
                print("I/O error")
        return self.medicatie

    def changeAnalize(self):
        """
        One Hot Encoding for the "Analize_prim_set" column.
        :return:
        """
        dan = {}
        indx = 0
        self.analize = dict()
        for record in self.df.Analize_prim_set:
            analz = record
            self.analize[indx] = analz
            if record is not np.NAN:
                record = record.replace("- HS * ", "")
                analize_list = record.split(" || ")
                analz = analize_list
                self.analize[indx] = analz
                for analiza in analize_list:
                    analiza_name, rest = analiza.split(" - ", 1)
                    result, ignore = rest.split(" ", 1)
                    result = result.replace("<", "")
                    analiza_name = analiza_name.replace(" ", "")
                    try:
                        result_int = float(result)
                        dan[analiza_name] = 1
                        try:
                            self.df[analiza_name][indx] = result_int
                        except:
                            self.df[analiza_name] = np.zeros(self.df.shape[0], dtype=float)
                            self.df[analiza_name][indx] = result_int
                            pd.to_numeric(self.df[analiza_name])
                    except:
                        pass
            indx += 1
        for key, value in dan.items():
            if self.df[key].sum() <= self.df.shape[0] * 0.2:
                self.df.drop([key], inplace=True, axis='columns')
        self.df.drop(['Analize_prim_set'], inplace=True, axis='columns')
        if not self.file_analize_opened:
            csv_file = "csv_analize.csv"
            try:
                with open(csv_file, 'w') as f:
                    for key in self.analize.keys():
                        f.write("%s,%s\n" % (key, self.analize[key]))
            except IOError:
                print("I/O error")
        return self.analize

    def changeComorbiditati(self):
        """
        Calculating the weight for each Illness
        :returns: Dictionary[illness code: weight]
        """
        self.comorbidityCounts = pd.DataFrame()
        self.comorbidityCounts = self.comorbidityCountsDataset()

        weight = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        forma_weight = {0: -1, 1: -0.5, 2: 0.25, 3: 0.5, 4: 1}
        total_count = self.df.stare_externare.value_counts().sum()
        count_forma = pd.DataFrame(self.df.stare_externare.value_counts()).to_dict()['stare_externare']
        for i in range(0, 5):
            weight[i] = forma_weight[i] * (1 - count_forma[i] / total_count)

        col_names = self.comorbidityCounts.index
        forma = {0: 'Vindecat', 1: 'Ameliorat', 2: 'Stationar', 3: 'Agravat', 4: 'Decedat'}
        comorbidityWeights = {}
        for names in col_names:
            comorbidityWeights[names] = 0
            for i in range(0, 5):
                comorbidityWeights[names] += self.comorbidityCounts[forma[i]][names] * weight[i]

        indx = 0
        self.comorb = dict()
        for row in self.df.Comorbiditati:
            if row is not np.NaN:
                comb_list = row.split(',')
                regspt = re.sub(r'(,)([A-Z])', r'@\2', row)
                regspt = re.sub(', ', ' ', regspt)
                regspt = re.split('@', regspt)
                comb_weight = 0
                self.comorb[indx] = regspt
                for comb in comb_list:
                    comb = comb.split(" ", 1)[0]
                    if comb in comorbidityWeights:
                        comb_weight += comorbidityWeights[comb]
                self.df["Comorbiditati"][indx] = float(comb_weight)
            else:
                self.df["Comorbiditati"][indx] = 0
            indx += 1

        self.df["Comorbiditati"].replace(0, np.NAN, inplace=True)
        self.df["Comorbiditati"] = self.df["Comorbiditati"].astype(float).interpolate(method='polynomial', order=2)

        self.df["Comorbiditati"] = self.df["Comorbiditati"].astype(float)
        if not self.file_comorb_opened:
            csv_file = "csv_comorb_weight.csv"
            try:
                with open(csv_file, 'w') as f:
                    for key in comorbidityWeights.keys():
                        f.write("%s,%s\n" % (key, comorbidityWeights[key]))
            except IOError:
                print("I/O error")
        if not self.file_comrb_opened:
            csv_filecm = "csv_comorbiditati.csv"
            try:
                with open(csv_filecm, 'w') as f:
                    for key in self.comorb.keys():
                        f.write("%s,%s\n" % (key, self.comorb[key]))
            except IOError:
                print("I/O error")
        return comorbidityWeights

    def changeDiagnos(self):
        """
        Tied to the function above
        :return:
        """
        indx = 0
        for row_int in self.df.Diag_pr_int:
            if row_int is not np.NaN:
                try:
                    self.df["Diag_pr_int"][indx] = self.comorbiditati[row_int]
                except:
                    self.df["Diag_pr_int"][indx] = np.NAN
            indx += 1
        self.df.Diag_pr_int = self.df.Diag_pr_int.astype(float).interpolate(method='polynomial', order=2)
        self.df.Diag_pr_int = self.df.Diag_pr_int.fillna(method='bfill')
        self.df["Diag_pr_int"] = self.df["Diag_pr_int"].astype(float)

        indx = 0
        for row_ext in self.df.Diag_pr_ext:
            if row_ext is not np.NaN:
                try:
                    self.df["Diag_pr_ext"][indx] = self.comorbiditati[row_ext]
                except:
                    self.df["Diag_pr_ext"][indx] = np.NAN
            indx += 1
        self.df["Diag_pr_ext"] = self.df["Diag_pr_ext"].astype(float).interpolate(method='polynomial', order=2)
        self.df["Diag_pr_ext"] = self.df["Diag_pr_ext"].fillna(method='bfill')
        self.df["Diag_pr_ext"] = self.df["Diag_pr_ext"].astype(float)

    def comorbidityCountsDataset(self):
        """
        Creates a dictionary with every illness and how many people had each type of severity.
        :returns: DataFrame
        """
        try:
            self.com_ext = self.df[["Comorbiditati", "stare_externare"]]
            comorbidityCountMatrix = {}
            comorbidityFile = open("text-comorbiditati.txt", "r")
            comorbidityNames = csv.reader(comorbidityFile)
            for row in comorbidityNames:
                # count, vindecat, ameliorat, stationar, agravat, decedat
                identifierUniqueCode = row[0].split(" ", 1)[0]
                comorbidityCountMatrix[identifierUniqueCode] = [0, 0, 0, 0, 0, 0]
                for comorbidityColumn, outcome in self.com_ext.itertuples(index=False):
                    if type(comorbidityColumn) is str and row[0] in comorbidityColumn:
                        comorbidityCountMatrix[identifierUniqueCode][0] = comorbidityCountMatrix[identifierUniqueCode][
                                                                              0] + 1
                        comorbidityCountMatrix[identifierUniqueCode][int(outcome) + 1] = \
                            comorbidityCountMatrix[identifierUniqueCode][int(outcome) + 1] + 1
            dictr = {}
            for key, value in comorbidityCountMatrix.items():
                dis = key.split(" ", 1)
                dictr[dis[0]] = value[5]
            comorbidityFile.close()

            self.comorbidityCounts = pd.DataFrame(comorbidityCountMatrix)
            self.comorbidityCounts = self.comorbidityCounts.transpose()
            self.comorbidityCounts.rename(
                columns={0: 'Count', 1: 'Vindecat', 2: 'Ameliorat', 3: 'Stationar', 4: 'Agravat', 5: 'Decedat'},
                inplace=True, errors="raise")
        except IOError:
            self.comorbidityCounts = pd.DataFrame()

        return self.comorbidityCounts

    def featureCross(self):
        '''
        Including new columns to give bonus meaning to important and relevant features.
        :return:
        '''
        self.df["DiagExt-Int"] = self.df["Diag_pr_ext"] - self.df["Diag_pr_int"]
        self.df["ZileMed"] = self.df["zile_ATI"] / self.df["Zile_spitalizare"]

    def prediction(self, prediction_data):
        # prediction_data = [age, sex, diagnos_int, spitalizare, ati, analize, id]

        age = prediction_data[0]
        gender = prediction_data[1]
        diag_init = prediction_data[2]
        zile_spit = prediction_data[3]
        zile_ati = prediction_data[4]
        analize = prediction_data[5]
        medication = prediction_data[6]
        comorb = prediction_data[7]

        print("\n\tPREDICTION\n")
        # newdataset = self.df.drop(["Sex", "Varsta", "Zile_spitalizare", "zile_ATI", "Diag_pr_int", 'Analize_prim_set', "Comorbiditati", "Diag_pr_ext", "stare_externare", "forma_boala"], axis=0, inplace=False)
        newdataset = self.df.drop(
            range(1, self.df.shape[0]), axis=0, inplace=False)
        newdataset = newdataset.drop(["stare_externare", "forma_boala"], axis='columns')
        for column in newdataset.columns:
            newdataset[column] = 0

        try:
            diag = float(self.comorbiditati[diag_init.split(" ")[0]])
        except KeyError:
            diag = 0

        dan = {}
        analize = analize.replace("- HS * ", "")
        analize_list = analize.split(" || ")
        for analiza in analize_list:
            try:
                analiza_name, rest = analiza.split(" - ", 1)
                result, ignore = rest.split(" ", 1)
                result = result.replace("<", "")
                analiza_name = analiza_name.replace(" ", "")
                result_int = float(result)
                try:
                    analiza_name = analiza_name.replace("*", "A").replace("(", "").replace(")", "").replace("%", "B")
                    analiza_name = analiza_name.replace("#", "C").replace("/", "").replace(",", '')
                    newdataset[analiza_name] = float(result_int)
                    dan[analiza_name] = 1
                except:
                    newdataset[analiza_name] = np.zeros(newdataset.shape[0], dtype=float)
                    newdataset[analiza_name] = result_int
                    pd.to_numeric(newdataset[analiza_name])
            except:
                print("Vai vai")
                pass


        med_list = medication.split("|| ")
        for med in med_list:
            med = med.replace(" ", "")
            try:
                med = med.replace("*", "A").replace("(", "").replace(")", "").replace("%", "B")
                med = med.replace("#", "C").replace("/", "").replace(",", '')
                newdataset[med] = float(1)
            except:
                newdataset[med] = np.zeros(newdataset.shape[0], dtype=float)
                newdataset[med] = 1
                pd.to_numeric(newdataset[med])

        newcomorb = 0
        comorb_list = comorb
        for comb in comorb_list:
            try:
                newcomorb += float(self.comorbiditati[comb.split(" ")[0]])
            except KeyError:
                newcomorb += 0

        newdataset["Comorbiditati"] = float(newcomorb)
        newdataset["Varsta"] = float(age)
        newdataset["Sex"] = float(gender)
        newdataset["Diag_pr_int"] = float(diag)
        newdataset["Diag_pr_ext"] = 0
        newdataset["Zile_spitalizare"] = float(zile_spit)
        newdataset["zile_ATI"] = float(zile_ati)

        newcols = [col for col in set(newdataset.columns).intersection(self.df.columns)]
        newdataset = newdataset[newcols]

        newdataset = newdataset.drop(["Unnamed: 0"], axis=1)
        newdataset = newdataset.drop(["FO"], axis=1)
        newdataset.rename(columns=lambda s: s.replace("*", "A").replace("(", "").replace(")", "").replace("%", "B")
                       .replace("#", "C").replace("/", "").replace(",", ''), inplace=True)
        newdataset.astype(np.float)
        print(newdataset)
        return newdataset


if __name__ == '__main__':
    pr = DataProcessing()
    pr.compareTestResults()
