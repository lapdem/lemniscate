import numpy as np
from util.activation_functions import activation_functions
from .forward_equations_o1 import *


class Theory:

    def __init__(self, hyperparameters):
        # activation function
        self.rho = activation_functions[hyperparameters["activation_function"]]
        self.lambda_b = hyperparameters["training"]["learning_rate"]
        self.lambda_w = hyperparameters["training"]["learning_rate"]
        # input layer width
        self.n_0 = hyperparameters["layerwidths"][0]
        self.L = len(hyperparameters["layerwidths"]) - 1
        # time
        self.t = 0

    def initialise_parameters(
        self, initial_parameter_config, random_number_generator=None
    ):
        self.C_b = initial_parameter_config["biases"]["variance"]
        self.C_w = initial_parameter_config["weights"]["variance"]

    def set_data(self, data):
        self.x = data[0]
        self.y = data[1]

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
        print("Eigenvalues of Theta")
        print(np.linalg.eigh(self.Theta))



    def set_training_data(self, training_data):
        #find the indices of the training data x in all data x
        self.training_indices = np.isin(self.x, training_data[0]).nonzero()[0]
        

    def evolve(self, steps=1):
        self.t += steps
        return self.compute_training_loss()

    def compute_output(self, output_indices=None):
        if output_indices is None:
            output_indices = range(len(self.x))
        return calculate_phi_bar(self.x, self.y, self.t, self.Theta, self.training_indices, output_indices)

    def compute_training_output(self):
        return self.compute_output(self.training_indices)
    
    def compute_loss(self, output_indices=None):
        if output_indices is None:
            output_indices = range(len(self.x))
        return float(np.mean(np.square(self.y[output_indices] - self.compute_output(output_indices))))
    
    def compute_training_loss(self):
        return self.compute_loss(self.training_indices)

    def get_NTK(self):
        return self.Theta
