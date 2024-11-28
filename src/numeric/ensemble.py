import numpy as np
from .network import Network

# import all Network implementations
import numeric.networkTF
import numeric.networkNP


class Ensemble:
    def __init__(self, ensemble_config) -> None:
        self.config = ensemble_config
        available_networks = Network.get_implementations()
        NetworkImplementation = available_networks[
            ensemble_config["network"]["implementation"]
        ]
        self.networks = [
            NetworkImplementation(ensemble_config["network"]["hyperparameters"])
            for i in range(self.config["number_of_networks"])
        ]

        initial_parameter_config = self.config["network"]["initial_parameters"]
        source = "source"
        source_generate = "generate"
        source_from_file = "from_file"

        if initial_parameter_config[source] == source_generate:
            random_number_generator = np.random.default_rng(
                initial_parameter_config[source_generate]["random_seed"]
            )
            for network in self.networks:
                network.initialise_parameters(
                    initial_parameter_config[source_generate], random_number_generator
                )
        elif initial_parameter_config[source] == source_from_file:
            raise NotImplementedError(
                f" source '{source_from_file}' is not implemented."
            )
        else:
            raise NotImplementedError(
                f" source '{initial_parameter_config[source]}' is not implemented."
            )

    def compute_outputs(self, inputs):
        return [network.compute_outputs(inputs) for network in self.networks]

    def set_training_data(self, training_data):
        for network in self.networks:
            network.set_training_data(training_data)

    # returns an list of losses
    def evolve(self, steps=1):
        return [network.evolve(steps) for network in self.networks]
