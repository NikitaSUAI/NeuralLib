import numpy as np
from params.NodeModel import my_sigmoid


class NN:
    def __init__(self, **params):
        # layers : [ size of first layer, size of second layer, ...]
        # len(layers) - amount of layers
        self.layers = params["layer"]
        # lr - learning rate
        self.lr = params["lr"]
        self.af = params["activation_func"]
        # weight between layers
        self.weight = [np.full((i, j), np.random.uniform(0, 1/np.sqrt(j)))
                       for i, j in zip(self.layers[:-1], self.layers[1:])]

    def train(self, signals:np.array, unswer:np.array):
        if len(signals) != self.layers[0]:
            raise BaseException("Всем пиздец")

    def query(self, signals:np.array):
        if len(signals) != self.layers[0]:
            raise BaseException("Всем пиздец")
        for w in self.weight:
            # dot product of weights and signals
            signals = self.af(np.dot(w, signals))
        return signals

    def get_param_in_json(self):
        pass


if __name__ == "__main__":
    nn = NN(layer=[2, 2, 2], lr=0.1, activation_func=my_sigmoid)
    print(nn.query(np.array([0.5, 0.5])))