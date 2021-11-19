from Backend.DataAnalysis import dataAnalysis as da
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder
import csv
import pickle

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class DataProcessing:
    def __init__(self, dataset):
        print("Processing the dataset")
        self.dataset = pd.DataFrame(dataset)
        self.df = pd.DataFrame(self.dataset)
        self.comorbiditati = None

        filename_dataset = "dataset_processed.csv"
        filename_comorb = "comorb_weight.csv"

        try:
            with open(filename_dataset, mode='r') as infile:
                print("Dataset saved from file " + filename_dataset)
                self.dataset = pd.read_csv(filename_dataset, parse_dates=True)
                self.df = pd.DataFrame(self.dataset)

            with open(filename_comorb, mode='r') as infile:
                print("Dictionary saved from file " + filename_comorb)
                reader = csv.reader(infile)
                self.comorbiditati = {rows[0]: rows[1] for rows in reader}

        except IOError:
            print("Files " + filename_dataset + " " + filename_comorb + " not found")
            self.dataset = dataset
            self.df = pd.DataFrame(self.dataset)

            self.boala = None

            self.changeMedicatie()
            self.changeAnalize()
            self.comorbiditati = self.changeComorbiditati()
            self.changeDiagnos()

            self.comorb = self.df["Comorbiditati"]
            self.exetern = self.df["stare_externare"]
            self.varsta = self.df["Varsta"]
            self.spitalizare = self.df["Zile_spitalizare"]
            self.diagnos_int = self.df["Diag_pr_int"]
            self.com_ext = self.df[["Comorbiditati", "stare_externare"]]

            self.df.to_csv(filename_dataset, header=True)

    def getDataset(self):
        return self.df

    def changeMedicatie(self):
        """
        One Hot Encoding for the "Medicatie" column.
        :return:
        """
        dm = {}
        indx = 0
        for record in self.df.Medicatie:
            med_list = str(record).split("||")
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

    def changeAnalize(self):
        """
        One Hot Encoding for the "Analize_prim_set" column.
        :return:
        """
        dan = {}
        indx = 0
        for record in self.df.Analize_prim_set:
            if record is not np.NAN:
                record = record.replace("- HS * ", "")
                analize_list = record.split(" || ")
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
                            self.df[analiza_name] = np.zeros(self.df.shape[0], dtype=int)
                            self.df[analiza_name][indx] = result_int
                            pd.to_numeric(self.df[analiza_name])
                    except:
                        pass
            indx += 1
        for key, value in dan.items():
            if self.df[key].sum() <= self.df.shape[0] * 0.2:
                self.df.drop([key], inplace=True, axis='columns')
        self.df.drop(['Analize_prim_set'], inplace=True, axis='columns')

    def changeComorbiditati(self):
        """
        Mambo Jumbo <- TO BE CHANGED, WE DON'T KNOW WHAT IS THIS FOR NOW
        :returns: Dictionary[illness code: weight]
        """
        self.boala = pd.DataFrame()
        self.boala = self.boalaDataset()

        weight = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        forma_weight = {0: -1, 1: -0.5, 2: 0.25, 3: 0.5, 4: 1}
        total_count = self.df.stare_externare.value_counts().sum()
        count_forma = pd.DataFrame(self.df.stare_externare.value_counts()).to_dict()['stare_externare']
        for i in range(0, 5):
            weight[i] = forma_weight[i] * (1 - count_forma[i] / total_count)

        col_names = self.boala.index
        forma = {0: 'Vindecat', 1: 'Ameliorat', 2: 'Stationar', 3: 'Agravat', 4: 'Decedat'}
        d = {}
        for names in col_names:
            d[names] = 0
            for i in range(0, 5):
                d[names] += self.boala[forma[i]][names] * weight[i]

        indx = 0
        for row in self.df.Comorbiditati:
            if row is not np.NaN:
                comb_list = row.split(",")
                comb_weight = 0
                for comb in comb_list:
                    comb = comb.split(" ", 1)[0]
                    if comb in d:
                        comb_weight += d[comb]
                self.df["Comorbiditati"][indx] = float(comb_weight)
            else:
                self.df["Comorbiditati"][indx] = 0
            indx += 1

        self.df["Comorbiditati"].replace(0, np.NAN, inplace=True)
        self.df["Comorbiditati"] = self.df["Comorbiditati"].astype(float).interpolate(method='polynomial', order=2)

        self.df["Comorbiditati"] = self.df["Comorbiditati"].astype(float)

        csv_file = "comorb_weight.csv"
        try:
            with open(csv_file, 'w') as f:
                for key in d.keys():
                    f.write("%s,%s\n" % (key, d[key]))
        except IOError:
            print("I/O error")

        return d

    def changeDiagnos(self):
        """
        Tied to the function above => CASCADE ON EDIT ABOVE FUNCTION
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

    def boalaDataset(self):
        """
        Creates a dictionary with every illness and how many people had each type of severity.
        :returns: DataFrame
        """
        try:
            self.com_ext = self.df[["Comorbiditati", "stare_externare"]]
            db = {}
            g = open("../DataAnalysis/text-comorbiditati.txt", "r")
            r = csv.reader(g)
            for row in r:
                # count, vindecat, ameliorat, stationar, agravat, decedat
                ds = row[0].split(" ", 1)[0]
                db[ds] = [0, 0, 0, 0, 0, 0]
                for cr, ext in self.com_ext.itertuples(index=False):
                    if type(cr) is str and row[0] in cr:
                        db[ds][0] = db[ds][0] + 1
                        db[ds][int(ext) + 1] = db[ds][int(ext) + 1] + 1
            dictr = {}
            for key, value in db.items():
                dis = key.split(" ", 1)
                dictr[dis[0]] = value[5]
            g.close()

            self.boala = pd.DataFrame(db)
            self.boala = self.boala.transpose()
            self.boala.rename(
                columns={0: 'Count', 1: 'Vindecat', 2: 'Ameliorat', 3: 'Stationar', 4: 'Agravat', 5: 'Decedat'},
                inplace=True, errors="raise")
        except IOError:
            self.boala = pd.DataFrame()

        return self.boala


if __name__ == '__main__':
    d = da.DataAnalysis("../TrainingData/dataset.csv")
    pr = DataProcessing(d.getDataset())
    print(pr.getDataset().head())
