from __future__ import division

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from Backend import dataProcessing as dp, dataAnalysis as da
import pandas as pd
import pickle
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
    f, ax = plt.subplots(figsize=(16, 8))
    plt.plot(x, y, '.r', markersize=0.3)
    '''Green triangles for additional data points'''
    if addn_x.any() and addn_y.any() is not None:
        plt.plot(addn_x, addn_y, 'xg', markersize=0.3)
        '''Blue regression line'''
        if reg_line:
            x_min_i = addn_x.argmin()
            x_max_i = addn_x.argmax()
            print(x_min_i, [addn_x[x_min_i], addn_y[x_min_i]])
            print(x_max_i, [addn_x[x_max_i], addn_y[x_max_i]])
            plt.plot([addn_x[x_min_i], addn_y[x_min_i]],
                     [addn_x[x_max_i], addn_y[x_max_i]], 'b-')
    plt.xticks(range(len(x[0])))
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2  # inch margin
    s = maxsize / plt.gcf().dpi * len(x[0]) + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    plt.show()


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivation(x):
    # return sigmoid(x)*(1-sigmoid(x))
    return x*(1-x)


class PredictionModel:
    def __init__(self, dataset):
        self.df = dataset.drop(["Unnamed: 0"], axis=1)
        self.df.rename(columns=lambda s: s.replace("*", "A").replace("(", "").replace(")", "").replace("%", "B").replace("#", "C").replace("/", "").replace(",", ''), inplace=True)

        '''Features and labels'''
        sc_X = MinMaxScaler()
        self.Y = self.df[["stare_externare"]]

        features = self.X = self.df.drop(["stare_externare", "forma_boala"], axis=1, inplace=False)
        features_std = features.std()
        features_mean = features.mean()
        self.X = (features - features_mean) / features_std
        self.X = pd.DataFrame(sc_X.fit_transform(self.X, ), columns=self.X.columns)
        normalize(self.X)

        oversample = SMOTE()
        self.X, self.Y = oversample.fit_resample(self.X, self.Y)

        rus = RandomUnderSampler(random_state=0)
        rus.fit(self.X, self.Y)

        self.X_train, X_rem, self.Y_train, Y_rem = train_test_split(self.X, self.Y, train_size=4 / 5, random_state=42)
        self.X_valid, self.X_test, self.Y_valid, self.Y_test = train_test_split(X_rem, Y_rem, test_size=1 / 2,
                                                                                random_state=42)
        # self.X_train = sc_X.fit_transform(self.X_train)
        # self.X_test = sc_X.transform(self.X_test)
        # self.X_valid = sc_X.transform(self.X_valid)

        '''Define model'''
        self.my_model = None

        self.w1 = np.zeros((20, self.X_train.shape[1]))
        # self.w1 = np.zeros(self.X_train.shape[1])
        self.w2 = np.zeros((20, 12))
        self.w3 = np.zeros((12, 1))
        self.output = np.zeros(self.Y_train.shape)

        self.learning_rate = 0.3
        self.epochs = 1000
        self.batch_size = 20

        ds = tf.data.Dataset.from_tensor_slices((dict(self.X), self.X))
        ds = ds.shuffle(buffer_size=len(self.X))

        feature_columns = []
        for column in self.X:
            feat = self.X[column]
            # feat_bound = list(np.arange(int(min(feat)), int(max(feat)), resolutionZ))
            feature_bound = tf.feature_column.numeric_column(key=column, dtype=tf.float64)
            feature_columns.append(feature_bound)

        print(feature_columns)

        self.my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

        # self.testSimpleModels()
        # odel = self.createANNModel()
        # self.trainANNModel(model, 15, 100)
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
            cv_results = cross_validate(model, self.X_train, self.Y_train, cv=kfold, scoring=scoring)
            clf = model.fit(self.X_train, self.Y_train)
            y_pred = clf.predict(self.X_test)
            print(name)
            print(classification_report(self.Y_valid, y_pred, target_names=target_names))

            results.append(cv_results)
            names.append(name)

            this_df = pd.DataFrame(cv_results)
            this_df['model'] = name
            dfs.append(this_df)

        final = pd.concat(dfs, ignore_index=True)

    def createANNModel(self, epochs, batch_size, learning_rate):
        """Create and compile a simple linear regression model."""
        # The following variables are the hyperparameters.
        # learning_rate = 0.01
        # epochs = 20
        # batch_size = 100

        # Most simple tf.keras models are sequential.
        model = tf.keras.models.Sequential()

        # define 10-fold cross validation test harness
        # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        # cvscores = []
        # testno = 0
        # for train, test in kfold.split(self.X_train, self.Y_train):
        # testno += 1
        # model = tf.keras.models.Sequential()

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
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.04),
                                        name='Hidden1'))

        model.add(tf.keras.layers.Dropout(rate=0.2))

        # Define the second hidden layer with 12 nodes.
        model.add(tf.keras.layers.Dense(units=12,
                                        activation='sigmoid',
                                        kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                        name='Hidden2'))

        # Define the output layer.
        model.add(tf.keras.layers.Dense(units=1,
                                        activation='sigmoid',
                                        name='Output'))

        # model.compile(loss='binary_crossentropy',
        # optimizer='adam', metrics=['accuracy'])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                      loss="mean_squared_error",
                      metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.BinaryAccuracy(),
                               tf.keras.metrics.Precision()])

        # Split the dataset into features and label.
        features = {name: np.array(value) for name, value in self.X_train.items()}
        label = np.array(self.Y_train)
        history = model.fit(x=features, y=label,
                            batch_size=batch_size,
                            epochs=epochs)

        # Get details that will be useful for plotting the loss curve.
        return_epochs = history.epoch
        hist = pd.DataFrame(history.history)
        rmse = hist["mean_squared_error"]
        # Plot the number of epoch and the mean squared error.
        plot_the_loss_curve(return_epochs, rmse)

        # Evaluate the model using the test data.
        sc_X = MinMaxScaler()
        print("\n Evaluate the linear regression model against the test set:")
        features_test = {name: np.array(value) for name, value in self.X_test.items()}
        label_test = np.array(self.Y_test)
        score = model.evaluate(x=features_test, y=label_test,
                               batch_size=batch_size, verbose=0)
        for i in range(0, 4):
            print("%s: %.2f%%" % (model.metrics_names[i], score[i] * 100))
        # cvscores.append(score[1] * 100)

        # print("%.2f%% (+/- %.2f%%)" % (np.float(np.mean(cvscores)), np.float(np.std(cvscores))))

        return model

    def trainANNModel(self, model, epochs, batch_size):
        """Feed a dataset into the model in order to train it."""

        # Split the dataset into features and label.
        features = {name: np.array(value) for name, value in self.X_train.items()}
        label = np.array(self.Y_train)
        X_valid_array = {name: np.array(value) for name, value in self.X_valid.items()}
        Y_valid_array = np.array(self.Y_valid)
        history = model.fit(x=features,  y=label,
                            validation_data=(X_valid_array, Y_valid_array),
                            batch_size=batch_size,
                            epochs=epochs)

        # Get details that will be useful for plotting the loss curve.
        return_epochs = history.epoch
        hist = pd.DataFrame(history.history)
        rmse = hist["mean_squared_error"]

        return return_epochs, rmse

    def testANNModel(self):
        # The following variables are the hyperparameters.
        learning_rate = 0.05
        epochs = 20
        batch_size = 100

        # Establish the model's topography.
        self.my_model = self.createANNModel(epochs, batch_size, learning_rate)

        # Train the model on the normalized training set.
        return_epochs, mse = self.trainANNModel(self.my_model, epochs, batch_size)
        plot_the_loss_curve(return_epochs, mse)

        sc_X = MinMaxScaler()
        print("\n Evaluate the linear regression model against the test set:")
        features_test = {name: np.array(value) for name, value in self.X_test.items()}
        label_test = np.array(self.Y_test)
        self.my_model.evaluate(x=features_test, y=label_test, batch_size=batch_size)

        validation = {name: np.array(value) for name, value in self.X_valid.items()}
        validation_labels = np.array(self.Y_valid).transpose()[0]
        prediction = self.my_model.predict(validation)
        predicted_vals = np.array(prediction)
        valid_data_array = self.X_valid.to_numpy()
        visualize_data(valid_data_array, validation_labels, valid_data_array, predicted_vals)

    def predict(self, values):
        # self.my_model = pickle.load(open('model.pkl', 'rb'))

        prediction = self.my_model.predict(values)
        predicted_vals = np.array(prediction)
        print(predicted_vals)
        predicted_vals_proba = self.my_model.predict_proba(values)
        print(predicted_vals_proba)

        # pickle.dump(self.model, open('model.pkl', 'wb'))

        return predicted_vals, predicted_vals_proba

    def feedForward(self, input):
        self.layer1 = sigmoid(np.dot(input, self.w1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.w2))
        self.output = sigmoid(np.dot(self.layer2, self.w3))
        return self.output

    def backprop(self, input, output):
        # find derivative of the loss function with respect to the weights
        d_w3 = np.dot(self.layer2.T, (2*(output - self.output)*sigmoid_derivation(self.output)*self.learning_rate))
        d_w2 = np.dot(self.layer1.T, (np.dot(2*(output - self.output)*sigmoid_derivation(self.output)*self.learning_rate, self.w3)*sigmoid_derivation(self.layer2)))
        d_w1 = np.dot(input.T, (np.dot(np.dot(2*(output - self.output)*sigmoid_derivation(self.output)*self.learning_rate, self.w3)*sigmoid_derivation(self.layer2), self.w2.T)*sigmoid_derivation(self.layer1)))

        # update weights
        self.w1 += d_w1
        self.w2 += d_w2
        self.w3 += d_w3

    def trainOnce(self, input, output):
        self.output = self.feedForward(input)
        self.backprop(input, output)

    def train(self):
        low_bound = 0
        high_bound = self.batch_size
        for epoch in range(self.epochs):
            batched_data = self.X_train[:][low_bound: high_bound]
            while high_bound <= self.X_train.shape[1]:
                batched_data = self.X_train[:][low_bound: high_bound]
                batched_output = self.Y_train[:][low_bound: high_bound]
                for row in range(batched_data):
                    self.trainOnce(batched_data[row], batched_output[row])
                low_bound = high_bound
                high_bound = high_bound + self.batch_size
                if high_bound < self.X_train.shape[1]:
                    for row in range(batched_data):
                        self.trainOnce(batched_data[row], batched_output[row])

    def clusteringData(self, filename):

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
    m.clusteringData('')
