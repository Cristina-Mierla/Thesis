from singleton import Singleton
from imports import *

import dataProcessing as dp, dataAnalysis as da

pd.options.display.max_rows = 10
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def plot_the_loss_curve(epochs, mse, fct):
    """Plot a curve of loss vs. epoch."""
    filename = 'modelPlots\\TrainingLoss'
    filename += datetime.now().strftime("_%Y-%m-%d_%H-%M")
    title = ''
    for f in fct:
        title += f + " "
    filename += '_' + title + '.png'

    plt.figure()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min() * 0.95, mse.max() * 1.03])
    plt.savefig(filename)
    # plt.show()
    plt.clf()


def visualize_data(y, pred_y):
    y = y.idxmax(axis=1).str.split('_', expand=True)[1].to_numpy()
    y = list(map(int, y))

    f, ax = plt.subplots(figsize=(5, 5))
    plt.plot(y, pred_y, 'Dr')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    # plt.show()
    plt.clf()

    filename = 'modelPlots\\PredictionComparison'
    filename += datetime.now().strftime("_%Y-%m-%d_%H-%M")
    filename += '.png'

    f, ax = plt.subplots(figsize=(20, 5))
    plt.plot(range(len(y)), y, 'pb')
    plt.plot(range(len(pred_y)), pred_y, 'dg')
    plt.tight_layout()
    plt.legend(['True', 'Predicted'], prop={'size': 10})
    plt.savefig(filename)
    # plt.show()
    plt.clf()


class PredictionModel(metaclass=Singleton):

    def __init__(self, dataset):
        super().__init__()
        self.df = dataset

        self.X = self.Y = self.X_train = self.X_test = self.X_valid = self.Y_train = self.Y_test = self.Y_valid = self.Y_test_array = self.Y_train_array = None
        self.my_model = self.my_feature_layer = None

        self.activations = ["elu", "exponential", "relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu"]

        self.initializeSets()
        try:
            # modelName = "allnewobj/trainedModel_05-20_17-43_0.95_1000_exponential_selu_selu"
            # modelName1 = "allnewobj/trainedModel_05-20_17-39_0.9_1000_exponential_tanh_selu"
            modelName = "bestnewobj/trainedModel_05-21_13-32_0.89_1000_exponential_selu_elu"
            self.my_model = tf.keras.models.load_model(modelName)

        except IOError or Exception:
            self.trainBestModel()

    def initializeSets(self):
        self.df = self.df.drop(["Unnamed: 0"], axis=1)
        self.df = self.df.drop(["FO"], axis=1)
        self.df.rename(columns=lambda s: s.replace("*", "A").replace("(", "").replace(")", "").replace("%", "B")
                       .replace("#", "C").replace("/", "").replace(",", ''), inplace=True)

        '''Features and labels'''
        sc_X = MinMaxScaler()
        self.Y = self.df[["stare_externare"]]

        features = self.X = self.df.drop(["stare_externare", "forma_boala"], axis=1, inplace=False)

        self.features_std = features.std()
        self.feature_max = features.transpose().max(axis=1).sort_index(key=lambda x: x.str.lower())
        self.feature_min = features.transpose().min(axis=1).sort_index(key=lambda x: x.str.lower())
        self.features_mean = features.mean()
        self.X = pd.DataFrame(sc_X.fit_transform(self.X), columns=self.X.columns)

        strategy = {0: 816, 1: 1224, 2: 612, 3: 408, 4: 1020}
        # strategy = {0: .2, 1: .3, 2: .15, 3: .1, 4: .25}
        # oversample = SMOTE(sampling_strategy=strategy, random_state=42)
        oversample = ADASYN(sampling_strategy=strategy, random_state=42)
        self.X, self.Y = oversample.fit_resample(self.X, self.Y)

        # rus = RandomUnderSampler(random_state=42)
        # self.X, self.Y = rus.fit_resample(self.X, self.Y)

        idx = np.random.RandomState(seed=42).permutation(self.X.index)
        self.X = self.X.reindex(idx)
        self.Y = self.Y.reindex(idx)

        print(self.X.head())
        print(self.X.shape)
        print(self.Y["stare_externare"].value_counts())

        self.Y = pd.get_dummies(self.Y["stare_externare"], prefix="Outcome")
        print(self.Y)

        self.X_train, X_rem, self.Y_train, Y_rem = train_test_split(self.X, self.Y, train_size=4 / 5, random_state=42)
        self.X_valid, self.X_test, self.Y_valid, self.Y_test = train_test_split(X_rem, Y_rem, test_size=1 / 3,
                                                                                random_state=42)

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

    def createANNModel(self, learning_rate, layers=2, n=None, activation_fct=None):
        """Create and compile a simple linear regression model."""
        # Most simple tf.keras models are sequential.
        if n is None:
            n = [12, 20]
        if activation_fct is None:
            activation_fct = ["relu", "sigmoid"]
        model = tf.keras.models.Sequential()

        # Add the layer containing the feature columns to the model.
        model.add(self.my_feature_layer)

        for i in range(layers):
            # Define the hidden layers
            name = 'Hidden' + str(i)
            model.add(tf.keras.layers.Dense(units=n[i],
                                            activation=activation_fct[i],
                                            kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01),      # Regularizer to apply a penalty on the layer's kernel
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            bias_regularizer=tf.keras.regularizers.l2(0.01),        # Regularizer to apply a penalty on the layer's bias
                                            name=name))

            # model.add(tf.keras.layers.Dropout(rate=0.3))
            model.add(tf.keras.layers.GaussianNoise(0.3))

        # Define the output layer.
        model.add(tf.keras.layers.Dense(units=5,
                                        # activation='sigmoid',
                                        kernel_initializer='he_normal',
                                        activation='softmax',
                                        name='Output',
                                        bias_initializer=tf.keras.initializers.Zeros()))

        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=learning_rate),
            loss='categorical_crossentropy',
            # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.CategoricalAccuracy(),
                     tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return model

    def trainANNModel(self, model, epochs, batch_size, activation_fct):
        """Feed a dataset into the model in order to train it."""

        # Split the dataset into features and label.
        model_name = "model with functions: "
        for function in activation_fct:
            model_name += function + " "
        print("Training " + model_name)
        features = {name: np.array(value) for name, value in self.X_train.items()}
        label = np.array(self.Y_train)
        X_valid_array = {name: np.array(value) for name, value in self.X_valid.items()}
        Y_valid_array = np.array(self.Y_valid)
        history = model.fit(x=features, y=label,
                            steps_per_epoch=len(features['Sex']) // batch_size + 1,
                            validation_data=(X_valid_array, Y_valid_array),
                            validation_steps=len(X_valid_array) // batch_size + 1,
                            batch_size=batch_size,
                            epochs=epochs)

        # Get details that will be useful for plotting the loss curve.
        return_epochs = history.epoch
        # hist = pd.DataFrame(history.history)
        hist = history.history
        # rmse = hist["mean_squared_error"]
        rmse = pd.Series(hist["loss"])
        plot_the_loss_curve(return_epochs, rmse, activation_fct)

        return return_epochs, rmse

    def testANNModel(self, batch_size=100, activation_fct=None):
        """After creating and training the model, it is evaluated using the testing set."""

        if activation_fct is None:
            activation_fct = ["relu", "sigmoid"]
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

        real = tf.math.argmax(validation_labels, axis=-1)
        pred = tf.math.argmax(prediction, axis=-1)
        matrix = confusion_matrix(real, pred)
        df_cm = pd.DataFrame(matrix, range(5), range(5))
        print(df_cm)
        plt.figure(figsize=(8, 8))
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="d")
        date = datetime.now().strftime("%m-%d_%H-%M")
        plt.savefig(f"modelPlots\\ConfusionMatrix_{date}_{str(activation_fct)}")
        plt.clf()

        validation_y = self.Y
        validation_x = {name: np.array(value) for name, value in self.X.items()}
        prediction_x = self.my_model.predict(validation_x)

        real_y = tf.math.argmax(validation_y, axis=-1)
        pred_x = tf.math.argmax(prediction_x, axis=-1)

        matrix_x = confusion_matrix(real_y, pred_x)
        df_cm_x = pd.DataFrame(matrix_x, range(5), range(5))
        print(df_cm_x)
        plt.figure(figsize=(8, 8))
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(df_cm_x, annot=True, annot_kws={"size": 16}, fmt="d")
        plt.savefig(f"modelPlots\\ConfusionMatrix_alldata_{date}_{str(activation_fct)}")
        plt.clf()

        precision = evaluation[3]
        recall = evaluation[4]
        F1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        print("F1: " + str(F1))

        evaluation.append(F1)
        evaluation.append(validation_error.mean())

        print("\nModel summary:")
        self.my_model.summary()
        for i in range(len(self.my_model.metrics_names)):
            print("%s: %.2f%%" % (self.my_model.metrics_names[i], evaluation[i] * 100))

        return evaluation

    def runPipeline(self, new_epochs=200, new_batch_size=100, layers=2, n=None, activation_fct=None):
        # The following variables are the hyperparameters.
        if activation_fct is None:
            activation_fct = ["relu", "sigmoid"]
        if n is None:
            n = [12, 20]
        learning_rate = 0.01
        epochs = new_epochs
        batch_size = new_batch_size

        # Establish the model's topography.
        self.my_model = self.createANNModel(learning_rate=learning_rate, layers=layers, n=n,
                                            activation_fct=activation_fct)

        # Train the model on the normalized training set.
        print("\nTraininng the model")
        self.trainANNModel(self.my_model, epochs, batch_size, activation_fct)

        # Evaluate the model
        evaluation = self.testANNModel(new_batch_size, activation_fct)

        # Save the model
        f1 = round(evaluation[-2], 2)
        name = str(f1) + "_" + str(epochs)
        for f in activation_fct:
            name += "_" + f
        self.saveModel(name)

        return evaluation

    def saveModel(self, name):
        print(f"Saving model: {name}")
        date = datetime.now().strftime("%m-%d_%H-%M")
        self.my_model.save(f'bestnewobj/trainedModel_{date}_{name}')

    def runAllModels(self):
        # 3 layers = 729 runs
        testingFile = pd.read_csv("csv_ANN_testing.csv")
        last_ind = int(testingFile["ind"].iloc[-1]) + 1
        print(last_ind)

        x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        hyper_layers = 3
        hyper_batch_size = 100
        hyper_epochs = 1000
        hyper_n = [100, 70, 20]
        for p in itertools.product(x, repeat=3):
            fct1 = self.activations[p[0]]
            fct2 = self.activations[p[1]]
            fct3 = self.activations[p[2]]
            fct = [fct1, fct2, fct3]
            row = [last_ind, hyper_batch_size, hyper_layers, hyper_n, hyper_epochs, fct]
            try:
                evaluation_result = self.runPipeline(new_batch_size=hyper_batch_size,
                                                     new_epochs=hyper_epochs,
                                                     layers=hyper_layers, n=hyper_n,
                                                     activation_fct=fct)
                row = row + evaluation_result

                with open("csv_ANN_testing.csv", 'a', newline='') as f_object:
                    print("New row: " + str(row))
                    writer_object = writer(f_object)
                    writer_object.writerow(row)

                last_ind += 1
            except:
                row = []

        f_object.close()

    def bestModels(self):
        testingFile = pd.read_csv("csv_ANN_testing.csv")
        last_ind = int(testingFile["ind"].iloc[-1]) + 1
        print(last_ind)

        # with open("csv_ANN_testing.csv", 'a', newline='') as f_object:
        #     row = ["indx", "batch_size", "hidden_layers", "nodes", "epochs", "activation_fct", "loss", "mse", "accuracy",
        #            "precision", "recall", "f1_score", "valid_error"]
        #     writer_object = writer(f_object)
        #     writer_object.writerow(row)

        hyper_layers = 3
        hyper_batch_size = 150
        hyper_epochs = 1000
        hyper_n = [100, 70, 20]

        # hyper_fct = [["relu", "exponential"], ["relu", "selu"], ["selu", "exponential"], ["tanh", "exponential"],
        #              ["softsign", "exponential"], ["softplus", "elu"], ["elu", "sigmoid"]]
        # hyper_fct = [["relu", "relu", "exponential"], ["relu", "selu", "selu"], ["selu", "relu", "exponential"], ["tanh", "relu", "exponential"],
        #              ["softsign", "relu", "exponential"], ["softplus", "selu", "elu"], ["elu", "relu", "sigmoid"], ["softplus", "elu", "sigmoid"], ["softsign", "selu", "sigmoid"]]
        hyper_fct = [['exponential', 'tanh', 'selu'], ['exponential', 'selu', 'selu'], ['exponential', 'selu', 'tanh'], ['exponential', 'elu', 'selu'], ['exponential', 'selu', 'elu']]

        err = []
        for fct in hyper_fct:
            row = [last_ind, hyper_batch_size, hyper_layers, hyper_n, hyper_epochs, fct]
            try:
                evaluation_result = self.runPipeline(new_batch_size=hyper_batch_size,
                                                     new_epochs=hyper_epochs,
                                                     layers=hyper_layers, n=hyper_n,
                                                     activation_fct=fct)

                # self.saveModel(f"{hyper_epochs}_{evaluation_result[-1]}_{fct}")

                err.append(evaluation_result)
                row = row + evaluation_result

                with open("csv_ANN_testing.csv", 'a', newline='') as f_object:
                    print("New row: " + str(row))
                    writer_object = writer(f_object)
                    writer_object.writerow(row)

                last_ind += 1
            except:
                row = []

        for result in err:
            print(result)

        f_object.close()

    def trainBestModel(self):
        hyper_layers = 3
        hyper_batch_size = 150
        hyper_epochs = 1000
        hyper_n = [100, 70, 20]
        # hyper_functions = ["relu", "selu", "selu"]
        hyper_functions = ["selu", "relu", "exponential"]
        self.runPipeline(new_batch_size=hyper_batch_size,
                         new_epochs=hyper_epochs,
                         layers=hyper_layers, n=hyper_n,
                         activation_fct=hyper_functions)
        # self.saveModel(f"{hyper_batch_size}_{hyper_functions}")

    def predict(self, values: pd.DataFrame):
        newvalues = values.transpose().sort_index(key=lambda x: x.str.lower())
        numarator = newvalues.subtract(self.feature_min, axis=0)
        numitor = self.feature_max.subtract(self.feature_min, axis=0)
        X_std = numarator.div(numitor, axis=0)
        array_vals = X_std.transpose()

        prediction_set = {name: np.array(value) for name, value in array_vals.items()}
        prediction = self.my_model.predict(prediction_set)
        predicted_vals = np.array(prediction)
        print(predicted_vals)
        return predicted_vals[0]

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
        # plt.show()
        plt.clf()

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
        # plt.show()
        plt.clf()

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
        # plt.show()
        plt.clf()

        filename2 += ".png"

        return filename1, filename2


if __name__ == '__main__':
    d = da.DataAnalysis("csv_dataset.csv")
    pr = dp.DataProcessing()
    # df = pd.read_csv("csv_small_sample.csv")
    # m = PredictionModel(df)
    m = PredictionModel(pr.getDataset())
    mm = PredictionModel(pr.getDataset())
    #
    # print(hex(id(m)))
    # print(hex(id(mm)))

    m.bestModels()
    # m.trainBestModel()

    # train10 = m.X_train[0:10]
    # for i in range(10):
    #     arr = train10.iloc[[i]]
    #     m.predict(arr)

    # m.runAllModels()

    # m.testSimpleModels()
