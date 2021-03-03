import numpy as np
import json
from params.NodeModel import my_sigmoid
from params.BPEModel import delta
import matplotlib.pyplot as plt

class NN:
    def __init__(self, **params):
        # layers : [ size of first layer, size of second layer, ...]
        # len(layers) - amount of layers
        self.layers = params["layer"]
        # lr - learning rate
        self.lr = params["lr"]
        self.af = params["activation_func"]
        # weight between layers
        self.weight = [np.random.uniform(0, 1/np.sqrt(j), (i, j))
                       for i, j in zip(self.layers[:-1], self.layers[1:])]

    def train(self, signals:np.array, unswer:np.array):
        errs = list()
        for signal, uns in zip(signals, unswer):
            err = self.query(signal) - uns
            self.backpropagation(err)
            errs.append((err**2).sum())
            if 0 < (err**2).sum() < 0.1:
                break
        plt.plot(range(len(errs)), errs)
        plt.show()


    def backpropagation(self, err):
        errors = [err, ]
        it = self.signals[::-1].__iter__()
        prev = next(it)
        for w in self.weight[::-1]:
            errors.append(np.dot(errors[-1], w.T))
            w -= self.lr * delta(errors[-1], prev := next(it), prev)
        return errors


    def query(self, signal:np.array):
        self.signals = [signal, ]
        if len(signal) != self.layers[0]:
            raise BaseException("Всем пиздец")
        for w in self.weight:
            # dot product of weights and signals
            signal = self.af(np.dot(signal, w))
            self.signals.append(signal)
        return signal

    def save(self):
        with open("params/params.json", "w") as f:
            params = [self.layers,
                      self.lr,
                      self.weight]
            # json.dump(self, f)
        print(self.weight)


if __name__ == "__main__":
    nn = NN(layer=[8, 8, 8, 8, 8, 8], lr=0.0001, activation_func=my_sigmoid)
    learn = [np.random.random(8) for i in range(10000)]
    uns = list()
    for i in learn:
        uns.append(1 - i)
    nn.train(learn, uns)
    nn.save()
    print(nn.query(np.array([0, 0, 0, 0, 1, 1, 1, 1])))