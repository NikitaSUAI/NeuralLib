import numpy as np
import json
# plt - need to take away
import matplotlib.pyplot as plt
from typing import List, Dict



class NN:
    """Class provides an opportunity to create and train your own neural
    network.

    """
    def __init__(self, **params):
        """Configure your own network with key-word args

        :param params: Options.
        :type params: Dict
        :param layer: list of layers like
                [ size of first layer, size of second layer, ...]
                len(layers) - amount of layers.
        :type layer: List[int]
        :param lr: learning rate.
        :type lr: float
        :param activation_func: pointer to custom activation function
        :type activation_func: def af(np.array)->np.array
        :raise: KeyError from kwargs.
        :return: NN object that ready to work.
        :rtype: NN object
        :todo:
            - add type hinting for kwargs
            - add loading from file
            - add kwargs for derivative of an activation function

        """
        self.layers = params["layer"]
        self.lr = params["lr"]
        self.af = params["activation_func"]
        # weight between layers
        self.weight = [np.random.uniform(0, 1/np.sqrt(j), (i, j))
                       for i, j in zip(self.layers[:-1], self.layers[1:])]

    def train(self, signals: np.array, answer: np.array):
        """ Methode use backpropagation of errors to train neural net

        :param signals: signal for input layer of neural net
        :type signals: np.array
        :param answer: answer for comparison
        :type answer: np.array
        :todo:
            - detach plotting

        """
        errs = list()
        for signal, uns in zip(signals, answer):
            err = uns - self.query(signal)
            self.backpropagation(err)
            errs.append((err**2).sum()/len(err))
            if 0 < (err**2).sum() < 0.0000001:
                break
        plt.plot(range(len(errs)), errs)
        plt.show()


    def backpropagation(self, err):
        """
        :param err: difference between correct answer and answer of nn
        :type err: np.array
        :return: matrix of errors for each layer
        :rtype: List[np.array]
        :todo:
            - take away derivative of an af
            - remove unnecessary comments
        """
        errors = [err, ]
        it = (self.signals[::-1]).__iter__()
        o = next(it)
        for w in self.weight[::-1]:
            errors.append(np.dot(w, errors[-1]))
            # w -= np.dot(self.lr * err * delta(prev:=next(it)), prev.T)
            # w -= self.lr * np.dot(err,
            #         np.dot(delta(prev :=
            # sig = self.af(np.dot(w, o))
            # sig *= (1 - sig)
            # ersig = errors[-1] * sig
            # dEdw = -1 * np.dot(ersig.reshape(ersig.size, 1), o.reshape(1, o.size))
            # w -= self.lr * dEdw
            # o = next(it)
            prev = o
            o = next(it)
            h = (errors[-1] * o * (1 - o))
            w += self.lr * np.dot(h.reshape(h.size, 1), prev.reshape(1, prev.size))
        return errors

    def query(self, signal:np.array):
        """ Predict answer

        :param signal: signal for input layer of neural net. Len of signal
            must equal len of first layer.
        :type signal: np.array
        :raise: BaseException.
        :return: predicted answer of neural net.
        :rtype: np.array
        :todo:
            - change exception
            - take away self.signals (PEP)
        """
        self.signals = [signal, ]
        if len(signal) != self.layers[0]:
            raise BaseException("ERR")
        for w in self.weight:
            # dot product of weights and signals
            signal = self.af(np.dot(signal, w))
            self.signals.append(signal)
        return signal

    def save(self, path="params/params.json"):
        """ Save params of nn to json file (not working yet)

        :param path: path to json file with params of neural network,
            by default is "params/params.json"
        :todo:
            - end this methode
        """
        with open(path, "w") as f:
            params = [self.layers,
                      self.lr,
                      self.weight]
            # json.dump(self, f)
        print(self.weight)

    def load(self, path="params/params.json"):
        """ Load params of nn from json file (not working yet)

        :param path: path to json file with params of neural network,
                by default is "params/params.json"
        :todo:
            - end this methode
        """
        ...
