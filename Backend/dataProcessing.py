from Backend import dataAnalysis as da
import numpy as np
import pandas as pd
import csv
import re

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class DataProcessing:
    def __init__(self, dataset):
        print("Processing the dataset")
        self.dataset = pd.DataFrame()
        self.df = pd.DataFrame(self.dataset)
        self.comorbiditati = None
        self.medicatie = None
        self.analize = None
        self.comorb = None

        filename_dataset = "csv_processedDataset.csv"
        filename_comorb = "csv_comorb_weight.csv"
        filename_comrb = "csv_comorbiditati.csv"
        filename_analize = "csv_analize.csv"
        filename_medicatie = "csv_medicatie.csv"

        try:
            with open(filename_dataset, mode='r') as infile:
                print("Dataset saved from file " + filename_dataset)
                self.dataset = pd.read_csv(filename_dataset, parse_dates=True)
                self.df = pd.DataFrame(self.dataset)

            with open(filename_comorb, mode='r') as infile:
                print("Dictionary saved from file " + filename_comorb)
                reader = csv.reader(infile)
                self.comorbiditati = {rows[0]: rows[1] for rows in reader}

            with open(filename_comrb, mode='r') as infile:
                print("Dictionary saved from file " + filename_comrb)
                reader = csv.reader(infile)
                self.comorb = {rows[0]: rows[1] for rows in reader}

            with open(filename_analize, mode='r') as infile:
                print("Dictionary saved from file " + filename_analize)
                reader = csv.reader(infile)
                self.analize = {rows[0]: rows[1] for rows in reader}

            with open(filename_medicatie, mode='r') as infile:
                print("Dictionary saved from file " + filename_medicatie)
                reader = csv.reader(infile)
                self.medicatie = {rows[0]: rows[1] for rows in reader}


        except IOError and FileNotFoundError:
            print("Files " + filename_dataset + " " + filename_comorb + " not found")
            self.dataset = dataset
            self.df = pd.DataFrame(self.dataset)

            self.boala = None

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

    def getDataset(self):
        return self.df

    def getMedicatie(self):
        return self.medicatie

    def getAnalize(self):
        return self.analize

    def getComorbiditati(self):
        return self.comorbiditati

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
                    if comb in d:
                        comb_weight += d[comb]
                self.df["Comorbiditati"][indx] = float(comb_weight)
            else:
                self.df["Comorbiditati"][indx] = 0
            indx += 1

        self.df["Comorbiditati"].replace(0, np.NAN, inplace=True)
        self.df["Comorbiditati"] = self.df["Comorbiditati"].astype(float).interpolate(method='polynomial', order=2)

        self.df["Comorbiditati"] = self.df["Comorbiditati"].astype(float)

        csv_file = "csv_comorb_weight.csv"
        try:
            with open(csv_file, 'w') as f:
                for key in d.keys():
                    f.write("%s,%s\n" % (key, d[key]))
        except IOError:
            print("I/O error")
        csv_filecm = "csv_comorbiditati.csv"
        try:
            with open(csv_filecm, 'w') as f:
                for key in self.comorb.keys():
                    f.write("%s,%s\n" % (key, self.comorb[key]))
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
            g = open("text-comorbiditati.txt", "r")
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

    def featureCross(self):
        '''
        Including new columns to give bonus meaning to important and relevant features.
        :return:
        '''
        self.df["DiagExt-Int"] = self.df["Diag_pr_ext"] - self.df["Diag_pr_int"]
        self.df["ZileMed"] = self.df["zile_ATI"] / self.df["Zile_spitalizare"]


if __name__ == '__main__':
    d = da.DataAnalysis("csv_dataset.csv")
    pr = DataProcessing(d.getDataset())
    print(pr.getDataset().head())
