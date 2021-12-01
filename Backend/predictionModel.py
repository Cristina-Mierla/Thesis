from __future__ import division
from Backend import dataProcessing as dp, dataAnalysis as da
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from sklearn.preprocessing import *
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from mlxtend.plotting import plot_decision_regions

pd.options.display.max_rows = 10
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def plot_the_loss_curve(epochs, mse):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min() * 0.95, mse.max() * 1.03])
    plt.show()


def visualize_data(x, y, addn_x=None, addn_y=None, reg_line=False):
    """Red dots for each point"""
    f, ax = plt.subplots(figsize=(8, 8))
    plt.plot(x, y, 'ro')
    '''Green triangles for additional data points'''
    if addn_x and addn_y is not None:
        plt.plot(addn_x, addn_y, 'g^')
        '''Blue regression line'''
        if reg_line:
            x_min_i = addn_x.argmin()
            x_max_i = addn_x.argmax()
            print(x_min_i, [addn_x[x_min_i], addn_y[x_min_i]])
            print(x_max_i, [addn_x[x_max_i], addn_y[x_max_i]])
            plt.plot([addn_x[x_min_i], addn_y[x_min_i]],
                     [addn_x[x_max_i], addn_y[x_max_i]], 'b-')
    plt.show()


class PredictionModel:
    def __init__(self, dataset):
        self.df = dataset.drop(["Unnamed: 0"], axis=1)
        self.df.rename(columns=lambda s: s.replace("*", "A").replace("(", "").replace(")", "").replace("%", "B").replace("#", "C").replace("/", "").replace(",",''), inplace=True)

        '''Features and labels'''
        sc_X = MinMaxScaler()
        self.Y = self.df[["stare_externare"]]

        features = self.X = self.df.drop(["stare_externare", "forma_boala"], axis=1, inplace=False)
        # features_std = features.std()
        # features_mean = features.mean()
        # self.X = (features - features_mean) / features_std
        #column_names = list(map(lambda x: x.replace('*', ''), self.X.columns.tolist()))
        self.X = pd.DataFrame(sc_X.fit_transform(self.X, ), columns=self.X.columns)
        normalize(self.X)

        oversample = SMOTE()
        # self.X, self.Y = oversample.fit_resample(self.X, self.Y)

        rus = RandomUnderSampler(random_state=0)
        rus.fit(self.X, self.Y)

        self.X_train, X_rem, self.Y_train, Y_rem = train_test_split(self.X, self.Y, train_size=4 / 5, random_state=42)
        self.X_valid, self.X_test, self.Y_valid, self.Y_test = train_test_split(X_rem, Y_rem, test_size=1 / 2,
                                                                                random_state=42)
        # self.X_train = sc_X.fit_transform(self.X_train)
        # self.X_test = sc_X.transform(self.X_test)
        # self.X_valid = sc_X.transform(self.X_valid)

        ds = tf.data.Dataset.from_tensor_slices((dict(self.X), self.X))
        ds = ds.shuffle(buffer_size=len(self.X))
        #print(ds)

        feature_columns = []
        for column in self.X:
            feat = self.X[column]
            # feat_bound = list(np.arange(int(min(feat)), int(max(feat)), resolutionZ))
            feature_bound = tf.feature_column.numeric_column(key=column, dtype=tf.float64)
            feature_columns.append(feature_bound)

        self.my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

        self.testANNModel()

    def getFeatures(self):
        return self.X

    def getLabels(self):
        return self.Y

    def testSimpleModels(self):
        dfs = []

        models = [('LOGREG', LogisticRegression()), ('RF', RandomForestClassifier()), ('KNN', KNeighborsClassifier()),
                  ('SVM', SVC()), ('TREECLASS', DecisionTreeClassifier()), ('ADA', AdaBoostClassifier()),
                  ('GNB', GaussianProcessClassifier()), ('MLP', MLPClassifier())]

        # training and testing the above models
        results = []
        names = []
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

        # target_names = ['Usor', 'Moderat', 'Sever'] # forma boala
        # target_names = ['Vindecat', 'Ameliorat', 'Stationar', 'Decedat']
        target_names = ['Vindecat', 'Ameliorat', 'Stationar', 'Agravat', 'Decedat']  # stare externare

        for name, model in models:
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=90210)
            cv_results = cross_validate(model, self.X_train, self.y_train, cv=kfold, scoring=scoring)
            clf = model.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            print(name)
            print(classification_report(self.y_valid, y_pred, target_names=target_names))

            results.append(cv_results)
            names.append(name)

            this_df = pd.DataFrame(cv_results)
            this_df['model'] = name
            dfs.append(this_df)

        final = pd.concat(dfs, ignore_index=True)

    def createANNModel(self, my_learning_rate):
        """Create and compile a simple linear regression model."""
        # Most simple tf.keras models are sequential.
        model = tf.keras.models.Sequential()

        # Add the layer containing the feature columns to the model.
        model.add(self.my_feature_layer)

        # Describe the topography of the model by calling the tf.keras.layers.Dense
        # method once for each layer. We've specified the following arguments:
        #   * units specifies the number of nodes in this layer.
        #   * activation specifies the activation function (Rectified Linear Unit).
        #   * name is just a string that can be useful when debugging.

        # Define the first hidden layer with 20 nodes.
        model.add(tf.keras.layers.Dense(units=20,
                                        activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.04),
                                        name='Hidden1'))

        # Define the second hidden layer with 12 nodes.
        model.add(tf.keras.layers.Dense(units=12,
                                        activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                        name='Hidden2'))

        # Define the output layer.
        model.add(tf.keras.layers.Dense(units=1,
                                        name='Output'))

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                      loss="mean_squared_error",
                      metrics=[tf.keras.metrics.MeanSquaredError()])

        return model

    def trainANNModel(self, model, epochs, batch_size):
        """Feed a dataset into the model in order to train it."""

        # Split the dataset into features and label.
        features = {name: np.array(value) for name, value in self.X_train.items()}
        label = np.array(self.Y_train)
        history = model.fit(x=features,  y=label, batch_size=batch_size, epochs=epochs)

        # Get details that will be useful for plotting the loss curve.
        epochs = history.epoch
        hist = pd.DataFrame(history.history)
        rmse = hist["mean_squared_error"]

        return epochs, rmse

    def testANNModel(self):
        # The following variables are the hyperparameters.
        learning_rate = 0.01
        epochs = 15
        batch_size = 100

        # Establish the model's topography.
        my_model = self.createANNModel(learning_rate)

        # Train the model on the normalized training set.
        epochs, mse = self.trainANNModel(my_model, epochs, batch_size)
        plot_the_loss_curve(epochs, mse)

        sc_X = MinMaxScaler()
        print("\n Evaluate the linear regression model against the test set:")
        features_test = {name: np.array(value) for name, value in self.X_test.items()}
        label_test = np.array(self.Y_test)
        my_model.evaluate(x=features_test, y=label_test, batch_size=batch_size)

        validation = {name: np.array(value) for name, value in self.X_valid.items()}
        validation_labels = np.array(self.Y_valid)
        prediction = my_model.predict(validation)
        predicted_vals = np.array(prediction)
        print(predicted_vals, '\n', validation_labels)
        print(validation)
        visualize_data(validation, validation_labels, validation, predicted_vals)


if __name__ == '__main__':
    d = da.DataAnalysis("csv_dataset.csv")
    pr = dp.DataProcessing(d.getDataset())
    m = PredictionModel(pr.getDataset())
    print(m.getFeatures())
    print(m.getLabels())
    # print(pr.getDataset().head())
