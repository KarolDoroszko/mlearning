import numpy as np
from matplotlib import pyplot as plt

class Estimator:
    def __init__(self, sample:list, domain_range:tuple=(-5,5), step: float = 0.01):
        self.sample = sample
        self.range = domain_range
        self.step = step
        self.width = self.find_width()

    def find_width(self):
        cardinality, std = len(self.sample), np.std(self.sample)
        return (4 / (3 * cardinality)) ** (0.2) * std

    def point_estimate(self, x):
        result = sum(np.exp(- ( x - xi )**2 / self.width**2)/ ( 2 * np.pi * self.width ** 2) for xi in self.sample) / len(self.sample)
        return result

    def estimate(self):
        result = list(map(lambda x: self.point_estimate(x), np.arange(*self.range, self.step)))
        plt.plot(np.arange(*self.range, self.step), result)
        plt.grid()
        plt.show()

if __name__ == '__main__':
    sample_size = 100
    z1 = np.random.normal(-2,0.5,sample_size)
    z2 = np.random.normal(5,0.3,sample_size)
    z = np.concatenate((z1,z2), axis = 0)

    estimator = Estimator(z, (-10,10), 0.05)
    estimator.estimate()


