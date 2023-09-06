import numpy as np
from dataclasses import dataclass
from pathlib import Path
from IFeedForwardNetwork import IFeedForwardNetwork

@dataclass
class FeedForwardNetwork(IFeedForwardNetwork):

    def backpropagation(self):
        pass

    @staticmethod
    def error_function(y,prediction):
        y = np.array([0 if i != y else 1 for i in range(0, 10)])
        return y - prediction
        # return np.linalg.norm(y - prediction) ** 2

    @staticmethod
    def nonlinear_function(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def nonlinear_derivative(x):
        return 1/(1+np.exp(-x)) * ( 1- 1/(1+np.exp(-x)) )


if __name__ == '__main__':
    data_path = Path(r"C:\Users\doros\Desktop\neural_network\neural-networks-and-deep-learning\data\mnist.pkl.gz")
    net_arch = [784,30,10]
    minibatch = 10
    learning_rate = 0.5
    epochs = 5
    net = FeedForwardNetwork(minibatch_size=minibatch, network_architecture=net_arch, epochs=epochs,
                             learning_rate=learning_rate, data_path=data_path)
    net.iterate_epochs()
    z = 0