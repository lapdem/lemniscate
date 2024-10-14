import numpy as np

activation_functions = {}
activation_function = lambda f: activation_functions.setdefault(f.__name__, f)

@activation_function
def tanh(input_values):
    return np.tanh(input_values)

@activation_function
def linear(input_values):
    return input_values