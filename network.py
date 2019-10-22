#!/usr/bin/env python3

"""
First exercise where whe will try to implement our first Neural Network that
recognises figures
"""

import random
import numpy as np

class Network(object): # signifie que Network est une sous-classe de object
    """
    Class representing our Neural Network
    """

    def __init__(self, sizes):
        """
        No biases for the input layer of neurons
        choice according to a gaussian distribution ~ N(0, 1)
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def backprop(self, x, y):
        """
        Returns a tuple (nabla_b, nabla_w) representing the gradient for the
        cost function with nabla_b and nabla_w layer by layer lists of arrays
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store the activation of each layer
        zs = [] # list to store the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backword pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """
        Returns the vector of partial derivatives of the cost function for the
        output activations
        """
        return output_activations - y


    def update_mini_batch(self, mini_batch, eta):
        """
        Updates the networks weights and biases according to the gradient
        descent using backpropagation on a single batch
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_w)]
        self.weights =[w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases =[b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def feedforward(self, a):
        """
        calculates the output of the network when a is the input
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Trains the Network with the SDG method by dividing the training_data into
        mini_batches and doing epochs on each of themself.
        The training_data is a list of tuples (x,y) where x is the input and y
        them expected outputself.
        etais the learning rate
        If test_data is provided then the Network will be tested againt this test_data
        after each epoch
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {} complete".format(j))

    def evaluate(self, test_data):
        """
        Returns the number of test inputs for which the neural network outputs
        the correct result
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)


def sigmoid(z):
    """
    Obvious implementation of the sigmoid function
    """
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(z)*(1-sigmoid(z))
