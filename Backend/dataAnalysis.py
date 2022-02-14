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
        print("Analyzing the dataset")
        self.df = pd.DataFrame()
        try:
            dataset = pd.read_csv(file)
            self.df = pd.DataFrame(dataset)
            # print(self.df.head())
            self.replaceColumns()
        except IOError:
            print("The file does not exist")

    def setDataset(self, dataset):
        print("Setting the dataset")
        self.df = dataset

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
        # print("\nDrop the columns that we are not going to use for the model")
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

    def getDataset(self):
        return self.df

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

    def pltReleaseState(self, filename):
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
        plt.savefig(filename)
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

        filename += ".png"

        return filename

    def groupAge(self, filename):
        ageFreq = self.df.groupby(pd.Grouper(key="Varsta"))["Sex"].count().sort_index(ascending=True)
        # print(ageFreq)
        n = (10 / 3) * np.log10(self.df["Varsta"].count()) + 1
        # print("Number of groups: " + str(n))
        grLen = (self.df["Varsta"].max() - self.df["Varsta"].min())/n
        # print("Length of the groups: " + str(grLen))

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
            # print(newrange + "\t" + str(newfreq))

        groupedAge = pd.DataFrame(newColumn, {"Freq"}).transpose()

        ax = sns.barplot(x=groupedAge.index, y=groupedAge.Freq, data=groupedAge)
        plt.plot(groupedAge.Freq, marker='s')
        ax.tick_params(axis='x', which='major', labelsize=8)
        plt.ylabel("Count of ages in each group")
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

        filename += ".png"

        return filename

    def clusteringData(self, filename, age, gender):
        # sns.set_palette("Purples_r")

        clusterData = self.df[self.df["Varsta"] == age]
        clusterData = clusterData[clusterData["Sex"] == gender]
        clusterData["forma_boala"] = clusterData["forma_boala"].astype(int)
        cols = ["Zile_spitalizare", "zile_ATI", "Comorbiditati", "stare_externare", 'forma_boala']
        print(clusterData.columns.get_loc("forma_boala"))
        print(clusterData.columns.tolist())
        print(clusterData.get("forma_boala", default="Name is not present"))

        pp = sns.pairplot(self.df[cols], palette="Set2",
                          diag_kind="kde", hue='forma_boala', markers=["s", "D", "^"])
        fig = pp.fig
        fig.subplots_adjust(top=0.93, wspace=0.3)
        t = fig.suptitle('Clustered data based on a given age and gender', fontsize=14)
        plt.savefig(filename)
        plt.show()

        filename += ".png"

        return filename


if __name__ == '__main__':
    d = DataAnalysis("csv_dataset.csv")
    d.clusteringData("", 53, 1)
    d.describe()
    # d.pltReleaseState()
