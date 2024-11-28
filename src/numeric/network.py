from abc import ABC, abstractmethod

"""
Abstract base class for a neural network.
Methods
-------
initialise_parameters(initial_parameter_config, random_number_generator) -> None
    Abstract method to initialise the network parameters.
compute_output(input: np.ndarray) -> np.ndarray
    Abstract method to compute the output of the network given an input.
evolve(training_data: Tuple[np.ndarray, np.ndarray]) -> float
    Abstract method to evolve the network one training step using the provided training data. 
    Returns the new loss after the training step.
"""
from typing import List, Tuple
import numpy as np


class Network(ABC):
    @abstractmethod
    def __init__(self, hyper_parameters):
        pass

    @staticmethod
    def get_implementations():
        implementations = Network.__subclasses__()
        return {impl.__name__: impl for impl in implementations}

    @abstractmethod
    def initialise_parameters(
        self, initial_parameter_config, random_number_generator
    ) -> None:
        pass

    @abstractmethod
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def compute_outputs(self, input: List[np.ndarray]) -> np.ndarray:
        pass

    @abstractmethod
    def set_training_data(self, training_data: Tuple[np.ndarray, np.ndarray]) -> None:
        pass

    @abstractmethod
    def evolve(self, steps: int = 1) -> float:
        pass
