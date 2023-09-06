import numpy as np
import sys
from matplotlib import pyplot as plt
sys.path.append(r'C:\Users\doros\Desktop\git\deep_learning\chapter1')
from gradient_descent import GradientDesc

#I am looking for the gradfollowing formula:
# F(x) = a1 * N(m1,s1) + a2 * N(m2,s2) + ... + an * N(mn,sn)
# Wariować by trzeba po 3 * n parametrach (a, m, s)
# We want to impoese:
# sum(ai) = 1, hence:
# ai = e**(ai) / (sum(e**ai))
# and s has to be symmetric and positively determined, hence:
# s = A * AT, for some A

class Gradient:
    def __init__(self, dim, components_number):
        self.dim = dim
        self.components_num = np.random.randn(0.1,1,components_number)
        self.coefficients = np.array([np.random.randn(0.1,1,dim) for i in range(components_number)])
        self.covariances = [np.random.randn(0.1,1,size = (dim, dim)) for i in range(components_number)]
        self.means = [np.random.randn(0.1,1,dim) for i in range(components_number)]
        self.errors = []
        def f(x,*args):
            coeff,covarinances = args[:len(self.components_num)], args[len(self.components_num):]
            splits = [self.dim for i in range(len(covarinances) / self.dim)]
            covarinances = iter(covarinances)
            cov = [np.array(list(islice(covarinances,split))).reshape((self.dim,self.dim)) for split in splits]
            result = sum(map(lambda a,c: a *x / np.sqrt(np.linalg.det(c) * (2 * np.pi) ** self.dim), zip(coeff,cov)))
            return result
        # self.initilaize_values()
        self.grad = GradientDesc(f,dim=(dim*dim + 1)* components_number, max_iter=50,
                                 dx=np.array([0.01 for i in range((dim*dim + 1)* components_number)]), alpha=.1)

    def initilaize_values(self):
        self.initialize_covariances()
        self.initialize_coefficients()

    def initialize_coefficients(self):
        normalization_factor = sum(np.exp(self.coefficients))
        self.coefficients = np.exp(self.coefficients) / normalization_factor

    def initialize_covariances(self):
        self.covariances = list(map(lambda matrix: matrix * np.transpose(matrix)), self.covariances)

    def make_iteration(self):
        pass

    def calculate_derivative(self):
        pass

    @staticmethod
    def criterium_check(value1, value2, threshold):
        if abs(value1 - value2) <= threshold:
            return True
        return False

    def run_gda(self):
        pass


if __name__ == '__main__':
    Z1 = np.random.multivariate_normal((10,10), np.array([[1,0],[0,1]]), 1000)
    Z2 = np.random.multivariate_normal((-5,-5), np.array([[1,0],[0,1]]), 1000)
    Z = np.concatenate((Z1,Z2),axis=0)
    plt.hist2d(list(Z[:,0]),list(Z[:,1]),bins=100)
    plt.show()

