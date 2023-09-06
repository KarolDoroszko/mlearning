import gzip
import pickle
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

@dataclass
class IFeedForwardNetwork(ABC):

    minibatch_size: int
    network_architecture: list
    epochs: int
    learning_rate: float
    data_path: Path
    train_val_sets_ratio: float = None
    weights = None
    biases = None
    activations = None
    minibatch_cost = None

    def __post_init__(self):
        self.initialize_weight()
        self.initialize_biases()
        self.minibatch_cost = []
        self.training_acc = []
        self.z = []
        self.a = []
        self.delta = []
        self.nabla_b = [np.zeros(b.shape) for b in self.biases]
        self.nabla_w = [np.zeros(w.shape) for w in self.weights]
        self.net_depth = len(self.network_architecture) - 2
        self.get_data()
        z =0

    @abstractmethod
    def backpropagation(self):
        raise NotImplementedError

    @staticmethod
    def error_function(y,prediction):
        y = np.array([0 if i !=y else 1 for i in range(0,10) ])
        return y-prediction

        # return np.linalg.norm(y-prediction) ** 2

    @staticmethod
    def accuracy(y, prediction):
        prediction = int(np.argmax(np.array(prediction)))
        # print(y,prediction)
        if y == prediction:
            return 1
        return 0

    @staticmethod
    def nonlinear_function(x):
        raise NotImplementedError

    @staticmethod
    def nonlinear_derivative(x):
        raise NotImplementedError

    def get_data(self):
        self.train_set, self.valid_set, self.test_set = \
            IFeedForwardNetwork.data_reader(self.data_path)

    def initialize_weight(self):
        self.weights = [np.random.standard_normal([layer_b,layer_f]) for layer_f, layer_b in
                        zip(self.network_architecture[:-1], self.network_architecture[1:])]

    def initialize_biases(self):
        self.biases = [np.random.standard_normal(layer) for layer in self.network_architecture[1:]]

    @staticmethod
    def data_reader(path: Path):
        print(path)
        with gzip.open(path, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            return train_set, valid_set, test_set

    def update_minibatch(self, starting_index: int, ending_index: int):
        # print('Updating next mininbatch...')
        acc = []
        for xi,yi in zip(self.train_set[0][starting_index:ending_index],
                         self.train_set[1][starting_index:ending_index]):
            a = xi
            self.a.append(a)
            for w, b in zip(self.weights, self.biases):
                z = np.dot(w,a) + b
                self.z.append(z)
                a = self.nonlinear_function(z)
                self.a.append(a)
            self.calculate_cost(yi,a)
            acc.append(self.accuracy(yi,a))
            nabla_w, nabla_b, delta = self.back_propagate()
            # print('-------------Shapes-----------')
            # for n, dn in zip(nabla_w, self.nabla_w):
            #     print(f'returned: {n.transpose().shape}, self: {dn.shape}')
            self.nabla_w = [dn.transpose() + n for dn, n in zip(nabla_w, self.nabla_w)]
            self.nabla_b = [dn + n for dn, n in zip(nabla_b, self.nabla_b)]
            self.delta = [dn + n for dn, n in zip(nabla_b, self.delta)]
            self.reset_a_z()
        self.training_acc += acc

    def check_validate(self):
        # print('Updating next mininbatch...')
        result = []
        for xi,yi in zip(self.valid_set[0],self.valid_set[1]):
            a = xi
            for w, b in zip(self.weights, self.biases):
                z = np.dot(w,a) + b
                a = self.nonlinear_function(z)
            result.append(self.accuracy(yi,a))
        return len(list(filter(lambda x: x==1, result))) / len(result)



    def train_model(self):
        minibatch_no = int(len(self.train_set[0]) / self.minibatch_size)
        print(f'Minibatch size is: {minibatch_no}')
        for minib_no in range(minibatch_no):
            self.feed_forward(minib_no)
            # self.print_progress()
            self.reset_cost()

    def print_progress(self):
        training_cost = len(list(filter(lambda x: x==1, self.training_acc))) / len(self.training_acc) * 100
        validation_cost = self.check_validate() * 100
        print(f'Error on training set ----------- {training_cost}%')
        print(f'Error on validation set --------- {validation_cost}%')

    def feed_forward(self, minibatchNo):
        # print('Starting training...')
        starting_index  = minibatchNo * self.minibatch_size
        ending_index = (minibatchNo + 1) * self.minibatch_size \
            if len(self.train_set[0]) > (minibatchNo + 1) * self.minibatch_size else len(self.train_set[0])
        minibatch_size = ending_index - starting_index

        self.update_minibatch(starting_index, ending_index)
        self.update_parameters(minibatch_size)

    def calculate_cost(self,y, prediction):
        self.minibatch_cost.append(self.error_function(y,prediction))

    def update_parameters(self, minibatch_size: int):
        self.weights = [w - self.learning_rate / minibatch_size * nw for w, nw in zip(self.weights, self.nabla_w)]
        self.biases = [b - self.learning_rate / minibatch_size * nb for b, nb in zip(self.biases, self.nabla_b)]
        self.reset_gradient_parameters()

    def reset_a_z(self):
        self.z = []
        self.a = []

    def reset_cost(self):
        self.minibatch_cost = []
        # self.training_acc = []

    def reset_gradient_parameters(self):
        self.delta = []
        self.nabla_b = [np.zeros(b.shape) for b in self.biases]
        self.nabla_w = [np.zeros(w.shape) for w in self.weights]

    def back_propagate(self):
        delta, nabla_w, nabla_b = [], [], []
        delta_L = self.minibatch_cost[-1] * self.nonlinear_derivative(self.z[-1])
        delta.append(delta_L)
        nabla_b.append(delta_L)
        nabla_w.append(np.outer(self.a[-2], delta_L))
        for el in range(self.net_depth-1,-1, -1):
            # print(el)
            delta_el = np.dot(self.weights[el+1].transpose(), delta[-1]) * self.nonlinear_derivative(self.z[el])
            nab_b = delta_el
            nab_w = np.outer(self.a[el], delta_el)
            delta.append(delta_el)
            nabla_b.append(nab_b)
            nabla_w.append(nab_w)
        nabla_w.reverse()
        nabla_b.reverse()
        delta.reverse()
        return nabla_w, nabla_b, delta

    def iterate_epochs(self):
        for i in range(self.epochs):
            self.training_acc
            print(f'\nEpoch --------- {i} ---------')
            self.train_model()
            self.print_progress()



def check(test_path: Path):
    print('Test of data reader')
    training,valid,test = IFeedForwardNetwork.data_reader(test_path)
    print(f'Length of data = {len(training[0])}, type = {type(training[0])} ')
    print(f'Length of data = {len(valid)}, type = {type(valid)} ')
    print(f'Length of data = {len(test)}, type = {type(test)} ')
    print('Test succesfull')
    return training

if __name__ == '__main__':
    start = datetime.now()
    test_data = Path(r"C:\Users\doros\Desktop\neural_network\neural-networks-and-deep-learning\data\mnist.pkl.gz")
    data = check(test_data)
    end = datetime.now()
    duration = end - start
    print(f'Duration of data reading was {duration}')