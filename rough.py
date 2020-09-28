import numpy as np
from abc import ABC, abstractmethod
from sklearn.datasets import make_blobs, make_circles, make_moons

class cost_function(ABC):
    @abstractmethod
    def cost(self):
        pass

class MSE(cost_function):
    def __init__(self,Y,pred_y,X):
        self.Y = Y
        self.X = X
        self.pred_y = pred_y
        self.n = Y.shape[0]
    def cost(self):
        cost = np.matmul((self.pred_y - self.Y).T, self.pred_y - self.Y) / (2 * self.n)
        d_cost = np.matmul(self.X.T, (self.pred_y - self.Y)) / self.n
        return np.squeeze(cost), d_cost

class CrossEntropy(cost_function):
    def __init__(self, Y,pred_y, X):
        self.Y = Y
        self.X = X
        self.pred_y = pred_y
        self.n = Y.shape[0]
    def cost(self):
        cost = -np.sum(self.Y * np.log(self.pred_y) + (1 - self.Y) * np.log(1 - self.pred_y)) / self.n
        d_cost = np.matmul(self.X.T, (self.pred_y - self.Y)) / self.n
        return cost, d_cost

class dataset_generator(ABC):
    @abstractmethod
    def generate_dataset(self):
        pass

class linear_data(dataset_generator):
    def __init__(self,m,n,sigma):
        self.m = m
        self.n = n
        self.sigma = sigma
    def generate_dataset(self):
        x_1 = np.random.randn(self.n, self.m)
        x_2 = np.ones((self.n, 1))
        x = np.concatenate((x_2, x_1), axis=1)
        e = np.random.normal(0, self.sigma, (self.n, 1))
        beta = np.random.rand(self.m + 1, 1)
        y = np.matmul(x, beta) + e
        return x,y,beta

class logistic_data(dataset_generator):
    def __init__(self,m,n,theta):
        self.m = m
        self.n = n
        self.theta = theta
    def generate_dataset(self):
        x1 = np.random.randn(self.n, self.m)
        x2 = np.ones((self.n, 1))
        x = np.concatenate((x2, x1), axis=1)
        beta = np.random.rand(self.m + 1, 1)
        y = 1/(1+np.exp(-(np.matmul(x,beta))))>0.5
        y = y.astype('float32')
        indices = np.random.choice(range(len(y)), int(self.theta * len(y)), replace=False)
        y[indices] = y[indices] == 0
        return x, y, beta


class clustering_data(dataset_generator):

    def generate_dataset(self):
        dataset = []

        # 4 make_blob datasets with diff parameters
        X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)
        dataset.append((X, y))
        X, y = make_blobs(n_samples=120, centers=3, n_features=2, random_state=1)
        dataset.append((X, y))
        X, y = make_blobs(n_samples=100, centers=4, n_features=2, random_state=2)
        dataset.append((X, y))
        X, y = make_blobs(n_samples=120, centers=4, n_features=3, random_state=3)
        dataset.append((X, y))

        # 3 make_circles dataset with diff parameters
        X, y = make_circles(n_samples=100, shuffle=True, noise=None, random_state=0, factor=0.8)
        dataset.append((X, y))
        X, y = make_circles(n_samples=120, shuffle=True, noise=0.05, random_state=1, factor=0.7)
        dataset.append((X, y))
        X, y = make_circles(n_samples=130, shuffle=True, noise=None, random_state=2, factor=0.3)
        dataset.append((X, y))

        # 3 make_moons dataset with diff parameters
        X, y = make_moons(n_samples=100, shuffle=True, noise=None, random_state=0)
        dataset.append((X, y))
        X, y = make_moons(n_samples=120, shuffle=True, noise=0.05, random_state=1)
        dataset.append((X, y))
        X, y = make_moons(n_samples=130, shuffle=True, noise=None, random_state=2)
        dataset.append((X, y))

        return dataset

class Optimizer_model(ABC):
    @abstractmethod
    def gradient(self):
        pass

class gradient_descent(Optimizer_model):
    def __init__(self, beta, alpha, d_cost):
        self.beta = beta
        self.d_cost = d_cost
        self.alpha = alpha
    def gradient(self):
        self.beta = self.beta - (self.alpha * self.d_cost)
        return self.beta

class algorithm(ABC):
    @abstractmethod
    def algo(self):
        pass

class linear_regression(algorithm):
    def __init__(self, X, Y, threshold, epochs, alpha, optimizer):
        self.X = X
        self.Y = Y
        self.threshold = threshold
        self.epochs = epochs
        self.alpha = alpha
        self.optimizer = optimizer

    def algo(self):
        prev_cost = float('inf')
        beta = np.random.rand(self.X.shape[1], 1)
        for i in range(self.epochs):
            pred_y = np.matmul(self.X, beta)
            object1 = MSE(self.Y, pred_y, self.X)
            cost, d_cost = object1.cost()
            if self.optimizer == "gradient_descent":
                object2 = gradient_descent(beta, self.alpha, d_cost)
            beta = object2.gradient()
            if abs(prev_cost - cost) <= self.threshold:
                break
            prev_cost = cost
        return cost, beta


class logistic_regression(algorithm):
    def __init__(self, X, Y, threshold, epochs, alpha, optimizer):
        self.X = X
        self.Y = Y
        self.threshold = threshold
        self.epochs = epochs
        self.alpha = alpha
        self.optimizer = optimizer

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def algo(self):
        prev_cost = float('inf')
        beta = np.random.rand(self.X.shape[1], 1)
        for i in range(self.epochs):
            pred_y = self.sigmoid(np.matmul(self.X, beta))
            object1 = CrossEntropy(self.Y, pred_y, self.X)
            cost, d_cost = object1.cost()
            if self.optimizer == "gradient_descent":
                object2 = gradient_descent(beta, self.alpha, d_cost)
            beta = object2.gradient()
            if abs(prev_cost - cost) <= self.threshold:
                break
            prev_cost = cost
        return cost, beta

# logistic_object=logistic_data(80,100,0.1)
# x,y,beta=logistic_object.generate_dataset()
# model_object=logistic_regression(x,y,0.00001,10000,.01,"gradient_descent")
# cost,learnt_beta=model_object.algo()
# print(cost,learnt_beta)

data_set = clustering_data()
ful = data_set.generate_dataset()
print(ful)
