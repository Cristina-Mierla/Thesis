from __future__ import division
import itertools
from csv import writer
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from mlxtend.evaluate import proportion_difference
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import *
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from Backend import dataProcessing as dp, dataAnalysis as da

pd.options.display.max_rows = 10
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def plot_the_loss_curve(epochs, mse, fct1, fct2):
    """Plot a curve of loss vs. epoch."""
    filename = 'modelPlots\\TrainingLoss'
    filename += datetime.now().strftime("_%Y-%m-%d_%H-%M")
    filename += '_' + fct1 + '_' + fct2 + '.png'

    plt.figure()
    plt.title(fct1 + ' - ' + fct2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min() * 0.95, mse.max() * 1.03])
    plt.savefig(filename)
    plt.show()


def visualize_data(y, pred_y):
    y = y.idxmax(axis=1).str.split('_', expand=True)[1].to_numpy()
    y = list(map(int, y))

    f, ax = plt.subplots(figsize=(5, 5))
    plt.plot(y, pred_y, 'Dr')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.show()

    filename = 'modelPlots\\PredictionComparison'
    filename += datetime.now().strftime("_%Y-%m-%d_%H-%M")
    filename += '.png'

    f, ax = plt.subplots(figsize=(20, 5))
    plt.plot(range(len(y)), y, 'pb')
    plt.plot(range(len(pred_y)), pred_y, 'dg')
    plt.tight_layout()
    plt.legend(['True', 'Predicted'], prop={'size': 10})
    plt.savefig(filename)
    plt.show()


class PredictionModel:
    def __init__(self, dataset):
        self.df = dataset

        self.X = self.Y = self.X_train = self.X_test = self.X_valid = self.Y_train = self.Y_test = self.Y_valid = self.Y_test_array = self.Y_train_array = None
        self.my_model = self.my_feature_layer = None

        self.activations = ["elu", "exponential", "relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu"]

        self.initializeSets()

    def initializeSets(self):
        self.df = self.df.drop(["Unnamed: 0"], axis=1)
        self.df = self.df.drop(["FO"], axis=1)
        self.df.rename(columns=lambda s: s.replace("*", "A").replace("(", "").replace(")", "").replace("%", "B")
                       .replace("#", "C").replace("/", "").replace(",", ''), inplace=True)

        '''Features and labels'''
        sc_X = MinMaxScaler()
        self.Y = self.df[["stare_externare"]]

        features = self.X = self.df.drop(["stare_externare", "forma_boala"], axis=1, inplace=False)
        features_std = features.std()
        features_mean = features.mean()
        self.X = (features - features_mean) / features_std
        self.X = pd.DataFrame(sc_X.fit_transform(self.X), columns=self.X.columns)
        normalize(self.X)

        oversample = SMOTE()
        self.X, self.Y = oversample.fit_resample(self.X, self.Y)

        rus = RandomUnderSampler(random_state=0)
        rus.fit(self.X, self.Y)

        # self.X = self.X.sample(frac=1)
        idx = np.random.permutation(self.X.index)
        self.X = self.X.reindex(idx)
        self.Y = self.Y.reindex(idx)

        print(self.X.head())
        print(self.X.shape)

        self.Y = pd.get_dummies(self.Y["stare_externare"], prefix="Outcome")
        print(self.Y)

        self.X_train, X_rem, self.Y_train, Y_rem = train_test_split(self.X, self.Y, train_size=4 / 5, random_state=42)
        self.X_valid, self.X_test, self.Y_valid, self.Y_test = train_test_split(X_rem, Y_rem, test_size=1 / 2,
                                                                                random_state=42)

        # self.X_train = sc_X.fit_transform(self.X_train)
        # self.X_test = sc_X.transform(self.X_test)
        # self.X_valid = sc_X.transform(self.X_valid)

        feature_columns = []
        for column in self.X:
            feature_bound = tf.feature_column.numeric_column(key=column, dtype=tf.float64)
            feature_columns.append(feature_bound)

        print(feature_columns)

        self.my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    def getFeatures(self):
        return self.X

    def getLabels(self):
        return self.Y

    def testSimpleModels(self):
        """Testing simple models"""
        dfs = []
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=90210)

        models = [
            ('LOGREG', LogisticRegression()),
            ('RF', RandomForestClassifier()), ('KNN', KNeighborsClassifier()),
            ('SVM', LinearSVC()), ('TREECLASS', DecisionTreeClassifier()),
            ('BAYES', GaussianNB()), ('MLP', MLPClassifier())]

        # reverse one hot encoding
        self.Y_test_array = self.Y_test.idxmax(axis=1).str.split('_', expand=True)[1]
        self.Y_train_array = self.Y_train.idxmax(axis=1).str.split('_', expand=True)[1]

        # training and testing the above models
        results = []
        names = []
        accuray_list = {}
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

        target_names = ['Vindecat', 'Ameliorat', 'Stationar', 'Agravat', 'Decedat']  # stare externare

        for name, model in models:
            cv_results = cross_validate(model, self.X_train, self.Y_train_array, cv=kfold, scoring=scoring)
            clf = model.fit(self.X_train, self.Y_train_array)
            y_pred = clf.predict(self.X_test)
            acc = accuracy_score(self.Y_test_array, y_pred)
            accuray_list[name] = acc
            print(name)
            print(classification_report(self.Y_test_array, y_pred, target_names=target_names))

            results.append(cv_results)
            names.append(name)

            this_df = pd.DataFrame(cv_results)
            this_df['model'] = name
            dfs.append(this_df)

        final = pd.concat(dfs, ignore_index=True)

        for m1, m2 in itertools.product(accuray_list, repeat=2):
            if m1 != m2:
                acc1 = accuray_list[m1]
                acc2 = accuray_list[m2]
                z, p = proportion_difference(acc1, acc2, n_1=len(self.Y_test_array))
                print(f"models: {m1}-{m2}\nz statistic: {z}, p-value: {p}")

    def createANNModel(self, learning_rate, n1=20, n2=12, activation_fct1="relu", activation_fct2="sigmoid"):
        """Create and compile a simple linear regression model."""
        # Most simple tf.keras models are sequential.
        model = tf.keras.models.Sequential()

        # Add the layer containing the feature columns to the model.
        model.add(self.my_feature_layer)

        # Define the first hidden layer with n1 nodes.
        model.add(tf.keras.layers.Dense(units=n1,
                                        activation=activation_fct1,
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.04),
                                        name='Hidden1'))

        model.add(tf.keras.layers.Dropout(rate=0.2))

        # Define the second hidden layer with n2 nodes.
        model.add(tf.keras.layers.Dense(units=n2,
                                        activation=activation_fct2,
                                        kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                        name='Hidden2'))

        # Define the output layer.
        model.add(tf.keras.layers.Dense(units=5,
                                        activation='sigmoid',
                                        # activation='softmax',
                                        name='Output'))

        model.compile(
            # optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            optimizer=tf.keras.optimizers.SGD(lr=learning_rate),
            # loss='binary_crossentropy',
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.BinaryAccuracy(),
                     tf.keras.metrics.Precision()])

        return model

    def trainANNModel(self, model, epochs, batch_size, function1, function2):
        """Feed a dataset into the model in order to train it."""

        # Split the dataset into features and label.
        print("Training %s - %s" % (function1, function2))
        features = {name: np.array(value) for name, value in self.X_train.items()}
        label = np.array(self.Y_train)
        X_valid_array = {name: np.array(value) for name, value in self.X_valid.items()}
        Y_valid_array = np.array(self.Y_valid)
        history = model.fit(x=features, y=label,
                            steps_per_epoch=len(features) // batch_size + 1,
                            validation_data=(X_valid_array, Y_valid_array),
                            validation_steps=len(X_valid_array) // batch_size + 1,
                            batch_size=batch_size,
                            epochs=epochs)

        # Get details that will be useful for plotting the loss curve.
        return_epochs = history.epoch
        hist = pd.DataFrame(history.history)
        rmse = hist["mean_squared_error"]
        plot_the_loss_curve(return_epochs, rmse, function1, function2)

        return return_epochs, rmse

    def testANNModel(self, new_epochs=200, new_batch_size=100, n1=20, n2=12, activation_fct1="relu",
                     activation_fct2="sigmoid"):
        """After creating and training the model, it is evaluated using the testing set."""

        # The following variables are the hyperparameters.
        learning_rate = 0.01
        epochs = new_epochs
        batch_size = new_batch_size

        # Establish the model's topography.
        self.my_model = self.createANNModel(learning_rate, n1, n2, activation_fct1, activation_fct2)

        # Train the model on the normalized training set.
        print("\nTraininng the model")
        self.trainANNModel(self.my_model, epochs, batch_size, activation_fct1, activation_fct2)

        print("\nEvaluate the linear regression model against the test set:")
        features_test = {name: np.array(value) for name, value in self.X_test.items()}
        label_test = self.Y_test
        evaluation = self.my_model.evaluate(x=features_test, y=label_test, batch_size=batch_size, verbose=0)

        validation = {name: np.array(value) for name, value in self.X_valid.items()}
        validation_labels = self.Y_valid
        prediction = self.my_model.predict(validation)
        final_prediction = []
        for line in prediction:
            index = np.where(line == line.max())[0][0]
            final_prediction.append(index)
        visualize_data(validation_labels, final_prediction)

        print("-------------------------------------------------------------------------------------------------------")
        print(prediction)
        err_list = validation_labels - prediction
        validation_error = err_list.mean()
        print("\nValidation mean error:\n" + str(validation_error))

        evaluation.append(validation_error.mean())

        print("\nModel summary:")
        self.my_model.summary()
        for i in range(0, 4):
            print("%s: %.2f%%" % (self.my_model.metrics_names[i], evaluation[i] * 100))

        return evaluation

    def runAllModels(self):
        testingFile = pd.read_csv("csv_ANN_testing.csv")
        last_ind = testingFile["ind"].iloc[-1] + 1

        x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        hyper_batch_size = 100
        hyper_epochs = 4000
        hyper_n1 = 100
        hyper_n2 = 30
        for p in itertools.product(x, repeat=2):
            fct1 = self.activations[p[0]]
            fct2 = self.activations[p[1]]
            row = [last_ind, hyper_batch_size, hyper_n1, hyper_n2, hyper_epochs, fct1, fct2]
            try:
                evaluation_result = self.testANNModel(hyper_epochs, hyper_batch_size, hyper_n1, hyper_n2, fct1, fct2)
                row = row + evaluation_result

                with open("csv_ANN_testing.csv", 'a', newline='') as f_object:
                    print("New row:" + str(row))
                    writer_object = writer(f_object)
                    writer_object.writerow(row)

                last_ind += 1
            except:
                row = []

        f_object.close()

    def predict(self, values):
        # self.my_model = pickle.load(open('model.pkl', 'rb'))
        # {name: np.array(value) for name, value in self.X_train.items()}

        array_vals = {name: np.array(value) for name, value in values.items()}
        prediction = self.my_model.predict(array_vals)
        predicted_vals = np.array(prediction)
        print(predicted_vals)
        predicted_vals_proba = self.my_model.predict_proba(array_vals)
        print(predicted_vals_proba)

        # pickle.dump(self.model, open('model.pkl', 'wb'))

        return predicted_vals, predicted_vals_proba

    def clusteringDataWithPCA(self, filename):

        filename1 = filename + '_PCA'
        filename2 = filename + '_KMeans'

        data = self.df.drop(['forma_boala', "stare_externare"], axis='columns').to_numpy()
        data = StandardScaler().fit_transform(data)
        target = self.df[['forma_boala', "stare_externare"]].to_numpy()

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(data)
        principalDf = pd.DataFrame(data=principalComponents,
                                   columns=['principal component 1',
                                            'principal component 2'])
        finalDf = pd.concat([principalDf, self.df[["stare_externare"]]], axis=1)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('Clustering using PCA', fontsize=20)
        targets = [0, 1, 2, 3, 4]
        colors = ['r', 'g', 'b', 'c', 'm']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf["stare_externare"] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c=color
                       , s=10)
        ax.legend(targets)
        ax.grid()
        plt.savefig(filename1)
        plt.show()

        filename1 += '.png'

        inertias = []

        # Creating 10 K-Mean models while varying the number of clusters (k)
        for k in range(1, 10):
            model = KMeans(n_clusters=k)

            # Fit model to samples
            model.fit(finalDf.iloc[:, :3])

            # Append the inertia to the list of inertias
            inertias.append(model.inertia_)

        plt.plot(range(1, 10), inertias, '-p', color='gold')
        plt.xlabel('number of clusters, k')
        plt.ylabel('inertia')
        plt.xticks(range(10))
        plt.show()

        model = KMeans(n_clusters=3)
        model.fit(finalDf.iloc[:, :2])

        labels = model.predict(principalDf.iloc[:, :2])
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(principalDf['principal component 1'],
                   principalDf['principal component 2'],
                   c=labels, s=10)
        ax.set_title("3-Means data clustering over PCA", fontsize=20)
        ax.grid()
        plt.savefig(filename2)
        plt.show()

        filename2 += ".png"

        return filename1, filename2


if __name__ == '__main__':
    d = da.DataAnalysis("csv_dataset.csv")
    pr = dp.DataProcessing()
    m = PredictionModel(pr.getDataset())


    # m.testSimpleModels()
