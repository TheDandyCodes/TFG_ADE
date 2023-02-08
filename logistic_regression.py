import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def init_parameters(self, X):
        self.w = np.zeros((1, X.shape[0]))
        self.b = 0

    def forward_propagation(self, X):
        z = np.dot(self.w, X) + self.b
        y_hat = 1/(1+np.exp(-z))
        return y_hat

    def backward_propagation(self, y_hat, y, X):
        dw = np.dot((y_hat - y), X.T) * (1/X.shape[1])
        db = np.sum((y_hat - y), axis=1, keepdims=True) * (1/X.shape[1])
        grads = {
            "dw":dw,
            "db":db
        }
        return grads

    def update_parameters(self, grads):
        self.w -= self.lr * grads["dw"]
        self.b -= self.lr * grads["db"]

    def predict(self, X):
        z = np.dot(self.w, X) + self.b
        y_hat = np.squeeze(1/(1+np.exp(-z)))
        class_prediction = [0 if y<=0.5 else 1 for y in y_hat]
        return class_prediction

    def accuracy(self, predictions, y):
        a = np.sum(predictions==y, axis=1)/y.shape[1]
        return a