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


class DataAnalysis:
    def __init__(self, file):
        try:
            dataset = pd.read_csv(file)
            self.df = pd.DataFrame(dataset)
            print(self.df.head())
            self.replaceColumns()
        except IOError:
            print("The file does not exist")

    def describe(self):
        """
        Important information about the data:
            - the number of lines x columns
            - the first 5 records of the dataset
            - the type of data in every row
            - how many null values are in every row
            - mean, min, max and the 4 important quartiles of the data
            - how many unique values are in a column
        :return:
        """
        print(self.df.head())
        print(self.df.shape)
        print("\nColumns of the data")
        print(self.df.columns)
        print("\nData types and nonnull values")
        print(self.df.info())
        print("\nNull values in the dataset")
        print(self.df.isnull().sum())
        print("\nDescribed dataset")
        print(self.df.describe().T)
        print("\nUnique values")
        print(self.df.nunique())

    def replaceColumns(self):
        """
        Removing redundant information.
        Replacing with Python NULL in the empty records.
        Making categorical data numerical.
        :return:
        """
        print("\nDrop the columns that we are not going to use for the model")
        self.df.drop(["AN", "precod", "FO", "Data_Examinare_Radiologie", "Radiologie", "rezultat_radiologie", "Proceduri", "Proceduri_Radiologie", "tip_externare", "unde_pleaca"], inplace=True, axis='columns')

        self.df.replace("NULL", np.NAN, inplace=True)
        self.df.replace("", np.NAN, inplace=True)
        self.df.replace("_", np.NAN, inplace=True)

        le = LabelEncoder()
        self.df["Sex"] = le.fit_transform(self.df["Sex"])
        self.df.stare_externare.replace(
            ("Vindecat", "Ameliorat", "Stationar", "AGRAVAT                                           ", "Decedat"),
            (0, 1, 2, 3, 4), inplace=True)
        self.df.forma_boala.replace(('1.USOARA', '2. MODERATA', '3.SEVERA', 'PERICLITAT TRANSFUZIONAL'), (1, 2, 3, np.NaN), inplace=True)

        self.df.forma_boala = self.df.forma_boala.fillna(self.df.forma_boala.median())

        self.changeMedicatie()
        self.changeAnalize()

    def changeMedicatie(self):
        """
        One Hot Encoding for the "Medicatie" column.
        :return:
        """
        d = {}
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
                d[med] = 1
            indx += 1
        for key, value in d.items():
            if self.df[key].sum() <= self.df.shape[0] * 0.2:
                self.df.drop([key], inplace=True, axis='columns')
        self.df.drop(['Medicatie'], inplace=True, axis='columns')

    def changeAnalize(self):
        """
        One Hot Encoding for the "Analize_prim_set" column.
        :return:
        """
        d = {}
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
                        d[analiza_name] = 1
                        try:
                            self.df[analiza_name][indx] = result_int
                        except:
                            self.df[analiza_name] = np.zeros(self.df.shape[0], dtype=int)
                            self.df[analiza_name][indx] = result_int
                            pd.to_numeric(self.df[analiza_name])
                    except:
                        pass
            indx += 1
        for key, value in d.items():
            if self.df[key].sum() <= self.df.shape[0] * 0.2:
                self.df.drop([key], inplace=True, axis='columns')
        self.df.drop(['Analize_prim_set'], inplace=True, axis='columns')

    def comorbiditati_columns(self):
        """
        Mambo Jumbo <- TO BE CHANGED, WE DON'T KNOW WHAT IS THIS FOR NOW
        :returns: Dictionary[illness code: weight]
        """
        self.boala = pd.DataFrame()
        self.boala = self.boala_dataset()

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

    def diagnos_columns(self):
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

    def boala_dataset(self):
        """
        Creates a dictionary with every illness and how many people had each type of severity.
        :returns: DataFrame
        """
        self.com_ext = self.df[["Comorbiditati", "stare_externare"]]
        d = {}
        g = open("text-comorbiditati.txt", "r")
        r = csv.reader(g)
        for row in r:
            # count, vindecat, ameliorat, stationar, agravat, decedat
            ds = row[0].split(" ", 1)[0]
            d[ds] = [0, 0, 0, 0, 0, 0]
            for cr, ext in self.com_ext.itertuples(index=False):
                if type(cr) is str and row[0] in cr:
                    d[ds][0] = d[ds][0] + 1
                    d[ds][int(ext) + 1] = d[ds][int(ext) + 1] + 1
        dictr = {}
        for key, value in d.items():
            dis = key.split(" ", 1)
            dictr[dis[0]] = value[5]
        g.close()

        self.boala = pd.DataFrame(d)
        self.boala = self.boala.transpose()
        self.boala.rename(
            columns={0: 'Count', 1: 'Vindecat', 2: 'Ameliorat', 3: 'Stationar', 4: 'Agravat', 5: 'Decedat'},
            inplace=True, errors="raise")
        return self.boala

    def pltColumnDistribution(self):
        sns.set_palette("Purples_r")

        plt.subplots(figsize=(13, 5))
        sns.countplot(data=self.df, x='Zile_spitalizare')
        plt.xticks(rotation=90)
        plt.xlabel("Days spent in the hospital")
        plt.title("Distribution of days spent in the hospital")
        plt.show()

        sns.kdeplot(self.df.stare_externare, data=self.df, shade=True, hue='Sex', label=['female', 'man'])
        plt.xlabel("Release state of the patients")
        plt.ylabel("Count")
        plt.title("Distribution of release state based on gender")
        plt.show()

        sns.kdeplot(self.df.Varsta, data=self.df, shade=True, hue='Sex', label=['female', 'man'])
        plt.title("Distribution of age based on gender")
        plt.show()

        sns.kdeplot(self.df.Zile_spitalizare, data=self.df, shade=True, hue='Sex', label=['female', 'man'])
        plt.xlabel("0-Curred, 1-Improved, 2-Stationary, 3-Worsted, 4-Deceased")
        plt.ylabel("Age")
        plt.title("Distribution of release states based on gender")
        plt.show()

        self.df["Sex"].value_counts().plot(kind="barh")
        plt.xticks(np.arange(0, 1000, step=100))
        plt.ylabel("0 - Female\n1 - Male")
        plt.xlabel("Count")
        plt.title("Count of each gender")
        plt.show()

        ax = sns.countplot(x=self.df["Sex"], data=self.df)
        for p in ax.patches:
            ax.annotate(f'\n{p.get_height()}', (p.get_x() + 0.2, p.get_height()),
                        ha='center', va='top', color='white', size=16)
        plt.xlabel('0 - Female    1 - Male')
        plt.title("Count of each gender")
        plt.show()

        plt.subplots(figsize=(12, 5))
        sns.distplot(self.df["Varsta"], bins=25, kde=True, rug=False)
        plt.title('Distribution of age')
        plt.show()

        plt.subplots(figsize=(12, 5))
        sns.distplot(self.df["Zile_spitalizare"], bins=70, kde=True, rug=False)
        plt.title('Distribution of hospitalization days')
        plt.xlabel("Hospitalization days")
        plt.show()

        with sns.axes_style('white'):
            sns.jointplot(x="Varsta", y='Zile_spitalizare', data=self.df, kind='hex')
            plt.show()

        sns.stripplot(data=self.df, x='stare_externare', y='Varsta', jitter=True, marker='.')
        plt.xlabel("Release State")
        plt.ylabel("Age")
        plt.show()

        f = sns.FacetGrid(self.df, col="stare_externare")
        f.map(plt.hist, "Zile_spitalizare")
        plt.xlim(0, 45)
        plt.show()

        with sns.color_palette("Purples"):
            f = sns.FacetGrid(self.df, col="forma_boala", hue="stare_externare")
            f.map(plt.scatter, "Zile_spitalizare", "zile_ATI", alpha=0.5, marker='.')
            f.add_legend()
            plt.xlim(0, 70)
            plt.show()

        g = sns.PairGrid(self.df, vars=["Zile_spitalizare", "zile_ATI", "Varsta"], hue="forma_boala")
        # g.map(plt.scatter)
        # g.map_diag(sns.histplot)
        # g.map_offdiag(sns.scatterplot)
        g.map_upper(sns.scatterplot, size=self.df["Sex"], alpha=0.5)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot)
        g.add_legend(title="", adjust_subtitles=True)
        plt.show()

    def pltReleaseState(self):
        sns.set_palette("Purples_r")

        data_vindecat = self.df[self.df["stare_externare"] == 0]
        data_decedat = self.df[self.df["stare_externare"] == 4]
        data_ameliorat = self.df[self.df["stare_externare"] == 1]
        varsta_vindecat = data_vindecat["Varsta"]
        varsta_decedat = data_decedat["Varsta"]
        varsta_ameliorat = data_ameliorat["Varsta"]
        spitalizare_vindecat = data_vindecat["Zile_spitalizare"]
        spitalizare_decedat = data_decedat["Zile_spitalizare"]
        spitalizare_ameliorat = data_ameliorat['Zile_spitalizare']
        ati_vindecat = data_vindecat["zile_ATI"]
        ati_ameliorat = data_ameliorat["zile_ATI"]
        ati_decedat = data_decedat["zile_ATI"]

        plt.plot(varsta_vindecat, spitalizare_vindecat, ".")
        plt.plot(varsta_ameliorat, spitalizare_ameliorat, ".", alpha=0.5)
        plt.plot(varsta_decedat, spitalizare_decedat, ".", alpha=0.5)
        plt.xlabel('Age')
        plt.ylabel('Hospitalization days')
        plt.legend(["Cured", "Improved", "Deceased"])
        plt.title('Hospitalization days per age based on release state')
        plt.show()

        plt.plot(varsta_vindecat, ati_vindecat, ".")
        plt.plot(varsta_ameliorat, ati_ameliorat, ".")
        plt.plot(varsta_decedat, ati_decedat, ".")
        plt.xlabel('Age')
        plt.ylabel('ICU days')
        plt.title("ICU days per age based on release state")
        plt.legend(["Cured", "Improved", "Deceased"])
        plt.show()

        fig = plt.figure(figsize=(9, 6))
        sns.histplot(data_decedat, x=data_decedat["Varsta"], y=data_decedat["Zile_spitalizare"], cmap='OrRd',
                     kde=True, label='Deceased', bins=45)
        sns.histplot(data_vindecat, x=data_vindecat["Varsta"], y=data_vindecat["Zile_spitalizare"], cmap='BuGn',
                     kde=True, label='Cured', bins=45, alpha=0.5)
        plt.ylabel("Hospitalization days")
        plt.xlabel("Age")
        patch1 = mpatches.Patch(color='green', label='Cured')
        patch2 = mpatches.Patch(color='orange', label='Deceased')
        plt.legend(handles=[patch1, patch2])
        plt.title("Hospitalization days per age based on release state")
        plt.show()

    def groupAge(self):
        ageFreq = self.df.groupby(pd.Grouper(key="Varsta"))["Sex"].count().sort_index(ascending=True)
        print(ageFreq)
        n = (10 / 3) * np.log10(self.df["Varsta"].count()) + 1
        print("Number of groups: " + str(n))
        grLen = (self.df["Varsta"].max() - self.df["Varsta"].min())/n
        print("Length of the groups: " + str(grLen))

        x1 = self.df["Varsta"].min()
        x2 = x1 + grLen
        newColumn = {}
        print("\nGroups and frequencies")
        for i in range(0, np.int(np.floor(n))):
            newrange = "[" + str(np.int(x1)) + ", " + str(np.int(x2)) + "]"
            newfreq = 0
            for loc in range(np.int(x1), np.int(x2)):
                try:
                    newfreq += ageFreq.iloc[loc-18]
                except IndexError:
                    pass
            x1 = x2
            x2 = x2 + grLen
            newColumn[newrange] = newfreq
            print(newrange + "\t" + str(newfreq))

        groupedAge = pd.DataFrame(newColumn, {"Freq"}).transpose()

        ax = sns.barplot(x=groupedAge.index, y=groupedAge.Freq, data=groupedAge)
        plt.plot(groupedAge.Freq, marker='s')
        ax.tick_params(axis='x', which='major', labelsize=8)
        plt.ylabel("Count of ages in each group")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    print("Analyzing the dataset")
    d = DataAnalysis("../TrainingData/dataset.csv")
    d.describe()
    # d.pltReleaseState()
