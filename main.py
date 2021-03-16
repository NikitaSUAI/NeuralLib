import numpy as np

from NN import NN
from params.NodeModel import my_sigmoid
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """ Quick start template.
    """
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
    nn = NN(layer=[1, 1, 1], lr=0.1, activation_func=my_sigmoid)
    learn = [np.array([np.random.rand(), ]) for i in range(100)]
    uns = [my_sigmoid(0.3*my_sigmoid(0.7*i)) for i in learn]
    for i, j in zip(learn, uns):
        print("{} -> {}".format(i, j))
    plt.scatter(learn, uns)
    plt.show()
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
