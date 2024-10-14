import numpy as np
from util.activation_functions import activation_functions
from .forward_equations import *


class Theory:

    def __init__(self, data, hyperparameters, bias_vaiance, weights_variance):
        # training inputs
        self.x = data[0]
        # training target outputs
        self.y = data[1]
        # activation function
        self.rho = activation_functions[hyperparameters["activation_function"]]
        self.lambda_b = hyperparameters["training"]["learning_rate"]
        self.lambda_w = hyperparameters["training"]["learning_rate"]
        self.C_b = bias_vaiance
        self.C_w = weights_variance
        # input layer width
        self.n_0 = hyperparameters["layerwidths"][0]
        # output layer index, layers are 1-indexed
        self.L = len(hyperparameters["layerwidths"])
        # time
        self.t = 0
        # Neural Tangent Kernel
        self.Theta = calculate_Theta(
            self.x,
            self.rho,
            self.lambda_b,
            self.lambda_w,
            self.C_b,
            self.C_w,
            self.n_0,
            self.L,
        )
        print("Theta")
        print(self.Theta)

    def evolve(self):
        self.t += 1

    def compute_training_ouputs(self):
        return calculate_phi_bar(self.x, self.y, self.t, self.Theta)
