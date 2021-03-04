import numpy as np
import json
from params.NodeModel import my_sigmoid
from params.BPEModel import delta
import matplotlib.pyplot as plt
from binary.convserter import B2D, D2B

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
            err = uns - self.query(signal)
            self.backpropagation(err)
            errs.append((err**2).sum()/len(err))
            # if 0 < (err**2).sum() < 0.1:
            #     break
        plt.plot(range(len(errs)), errs)
        plt.show()


    def backpropagation(self, err):
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

    # nn = NN(layer=[8, 6, 4, 2], lr=0.1, activation_func=my_sigmoid)
    # temp = lambda x: D2B(x)
    # magic = lambda x: [1, 0] if B2D(x) < 16 else [0, 1]
    # prepere = lambda x : np.array([ 0.9 if i == 1 else 0.1 for i in x])
    # learn = [temp(np.random.randint(100)) for _ in range(370000)]
    # uns = [magic(x) for x in learn]
    # nn.train(learn, uns)
    # nn.save()
    # print(nn.query(np.array([0, 0, 0, 0, 1, 1, 1, 1])))

    # nn = NN(layer=[8, 6, 4, 2], lr=0.01, activation_func=my_sigmoid)
    # temp = lambda x: D2B(x)
    # magic = lambda x: [1, 0] if B2D(x) > 15 else [0, 1]
    # prepere = lambda x : np.array([ 0.9 if i == 1 else 0.1 for i in x])
    # learn = [temp(np.random.randint(100)) for _ in range(100000)]
    # uns = [magic(x) for x in learn]
    # nn.train(learn, uns)
    # nn.save()
    # print(nn.query(np.array([0, 0, 0, 0, 1, 1, 1, 1])))

    # nn = NN(layer=[8, 1, 8], lr=0.01, activation_func=my_sigmoid)
    # f = lambda: [1 if np.random.rand() > 0.5 else 0 for _ in range(8)]
    # ref = lambda x: [0 if i > 0.5 else 1 for i in x]
    # learn = [np.array(f()) for _ in range(10000)]
    # uns = list()
    # for i in learn:
    #     uns.append(ref(i))
    # nn.train(learn, uns)
    # nn.save()
    # print(nn.query(np.array([0, 0, 0, 0, 1, 1, 1, 1])))
    #
    nn = NN(layer=[1, 1], lr=0.1, activation_func=my_sigmoid)
    learn = [np.array([np.random.rand(), ])for i in range(1000000)]
    uns = list()
    for i in learn:
        uns.append(i / 8)
    nn.train(learn, uns)
    nn.save()
    print(nn.query([0.8, ]))
    #
    # nn = NN(layer=[3, 3, 3], lr=0.3, activation_func=my_sigmoid)
    # learn = [np.array([np.random.rand(), ])for i in range(1000000)]
    # uns = list()
    # for i in learn:
    #     uns.append(i / 8)
    # nn.train(learn, uns)
    # nn.save()
    # print(nn.query([0.8, ]))