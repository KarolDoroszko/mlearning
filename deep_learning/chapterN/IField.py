import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class IFeedForwardNetwork(abc):

    minibatch_size: int
    network_architecture: np.array()
    training_set_ratio: float


    @abstractmethod
    def errof_function(self):
        raise NotImplementedError

    @abstractmethod
    def backpropagation(self):
        raise NotImplementedError

    @abstractmethod
    def nonlinear_function(self):
        raise NotImplementedError