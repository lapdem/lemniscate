import numpy as np
from .network import Network


class Ensemble:
    def __init__(self, ensemble_config) -> None:
        self.config = ensemble_config
        self.networks = [Network(ensemble_config["network"]["hyperparameters"]) for i in range(self.config["number_of_networks"])]

        initial_parameter_config = self.config["network"]["initial_parameters"]
        source = "source"
        source_generate = "generate"
        source_from_file = "from_file"

        if(initial_parameter_config[source] == source_generate):
            random_number_generator = np.random.default_rng(initial_parameter_config[source_generate]["random_seed"])
            for network in self.networks:
                network.initialise_parameters(initial_parameter_config[source_generate], random_number_generator)
        elif((initial_parameter_config[source] == source_from_file)):
            raise NotImplementedError(f" source '{source_from_file}' is not implemented.")
        else:
            raise NotImplementedError(f" source '{initial_parameter_config[source]}' is not implemented.")

    def compute_outputs(self, inputs):
        return [ [network.compute_output(input) for input in inputs]for network in self.networks]
    
    def evolve(self, training_data):
        for network in self.networks:
                network.evolve(training_data)