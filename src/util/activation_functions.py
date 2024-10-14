import numpy as np
from numba import jit

activation_functions = {}
activation_function = lambda f: activation_functions.setdefault(f.__name__, f)


@activation_function
@jit
def tanh(input_values):
    return np.tanh(input_values)


@activation_function
@jit
def linear(input_values):
    return input_values
