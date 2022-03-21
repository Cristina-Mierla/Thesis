import pandas as pd
from csv import writer
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.max_rows = 10
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivation(x):
    # return sigmoid(x)*(1-sigmoid(x))
    return x * (1 - x)


class CustomModel:
    def __init__(self):
        """Custom model"""
        self.w1 = self.w2 = self.w3 = self.output = self.learning_rate = self.epochs = self.batch_size = None

        self.X = self.Y = self.X_train = self.X_test = self.X_valid = self.Y_train = self.Y_test = self.Y_valid = self.Y_test_array = self.Y_train_array = None
        self.my_model = None

    def defineCustomModel(self):
        """Define model"""
        self.my_model = None

        self.w1 = np.zeros((20, self.X_train.shape[1]))
        # self.w1 = np.zeros(self.X_train.shape[1])
        self.w2 = np.zeros((20, 12))
        self.w3 = np.zeros((12, 1))
        self.output = np.zeros(self.Y_train.shape)

        self.learning_rate = 0.3
        self.epochs = 1000
        self.batch_size = 20

    def feedForward(self, input):
        self.layer1 = sigmoid(np.dot(input, self.w1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.w2))
        self.output = sigmoid(np.dot(self.layer2, self.w3))
        return self.output

    def backprop(self, input, output):
        # find derivative of the loss function with respect to the weights
        d_w3 = np.dot(self.layer2.T,
                      (2 * (output - self.output) * sigmoid_derivation(self.output) * self.learning_rate))
        d_w2 = np.dot(self.layer1.T, (
                np.dot(2 * (output - self.output) * sigmoid_derivation(self.output) * self.learning_rate,
                       self.w3) * sigmoid_derivation(self.layer2)))
        d_w1 = np.dot(input.T, (np.dot(
            np.dot(2 * (output - self.output) * sigmoid_derivation(self.output) * self.learning_rate,
                   self.w3) * sigmoid_derivation(self.layer2), self.w2.T) * sigmoid_derivation(self.layer1)))

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

