.. NeuralNetLib documentation master file, created by
   sphinx-quickstart on Tue Mar 16 18:13:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NeuralNetLib's documentation!
========================================

NeuralNetLib will support you create your own neural network.
You can choose an amount of layers, length of each layer, activation
function and learning rate.

This lib is written from the book "Make your own neural network by Tariq
Rashid"

**Use**:
    >>> # create neural net with 3 layers, each are corresponds to 1, 1, 1
    ... # learning rate  = 0.1
    ... # activation function is sigmoid
    ... # you can choose each of this params yourself
    ... nn = NN(layer=[1, 1, 1], lr=0.1, activation_func=my_sigmoid)
    ... # data for train our nn
    ... learn = [np.array([np.random.rand(), ]) for i in range(100)]
    ... uns = [my_sigmoid(0.3*my_sigmoid(0.7*i)) for i in learn]
    ... # train our nn
    ... nn.train(learn, uns)
    ... # and test it
    ... nn.query([0.8, ])
    ... # the answer will be between 0.53 and 0.55

Installation
-------------
    git clone https://github.com/NikitaSUAI/NeuralLib

    py -m pip install -r requirements.txt

Requirements
------------
* Python 3.9
* Numpy
* Matplotlib

License
-------
The project is licensed under the MIT license.

The User Guide
--------------

.. module:: NN
.. py:class:: NN.NN
.. autofunction:: NN.NN.__init__
.. autofunction:: NN.NN.train
.. autofunction:: NN.NN.backpropagation
.. autofunction:: NN.NN.query
.. autofunction:: NN.NN.save
.. autofunction:: NN.NN.load

.. module:: NodeModel
.. autofunction:: sigmoid
.. autofunction:: my_sigmoid

.. module:: BPEModel
.. autofunction:: delta