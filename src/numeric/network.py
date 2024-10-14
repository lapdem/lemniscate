import numpy as np
import util.data_generation as dg
from util.activation_functions import activation_functions


loss_functions = {}
loss_function = lambda f: loss_functions.setdefault(f.__name__, f)


@loss_function
def mean_square(actual, target):
    return (np.square(actual - target)).mean()


class Network:

    def __init__(self, hyper_parameters, random_number_generator=None) -> None:
        self.hyperparameters = hyper_parameters

    def _compute_activation(preactivation):
        pass

    def initialise_parameters(self, initial_parameter_config, random_number_generator):
        self.parameters = {}
        layerwidths = self.hyperparameters["layerwidths"]
        # no bias on input layer
        biases_sizes = layerwidths[1:]
        weights_sizes = list(zip(layerwidths[:-1], layerwidths[1:]))
        self.parameters["biases"] = [
            dg.generate_random_numbers(
                width,
                initial_parameter_config["biases"],
                random_number_generator=random_number_generator,
            )
            for width in layerwidths[1:]
        ]
        self.parameters["weights"] = [
            dg.generate_random_numbers(
                (first_width, second_width),
                initial_parameter_config["weights"],
                random_number_generator=random_number_generator,
            )
            for first_width, second_width in zip(layerwidths[:-1], layerwidths[1:])
        ]

    def compute_output(self, input):
        activation = input
        for i, (layer_biases, layer_weights) in enumerate(
            zip(self.parameters["biases"], self.parameters["weights"])
        ):
            preactivation = (
                np.dot(activation, layer_weights)
                * 1.0
                / np.sqrt(layer_weights.shape[0])
                + layer_biases
            )
            if i < (len(self.parameters["biases"]) - 1):
                activation = self.activation_function(preactivation)
            else:
                activation = self.output_activation_function(preactivation)
        return activation
    
    def evolve(self, training_data):
        learning_rate = self.hyperparameters["training"]["learning_rate"]
        delta = self.hyperparameters["training"]["numeric_delta"]

        original_loss = self.compute_loss(training_data)

        biases = self.parameters["biases"]
        new_biases = []
        for i, layer_biases in enumerate(biases):
            new_layer_biases = []
            for j, bias in enumerate(layer_biases):
                biases[i][j] += delta
                adjusted_loss = self.compute_loss(training_data)
                # reset bias
                biases[i][j] -= delta
                gradient = (adjusted_loss - original_loss) / delta
                new_layer_biases.append(biases[i][j] - learning_rate * gradient)
            new_biases.append(np.array(new_layer_biases))

        weights = self.parameters["weights"]
        new_weights = []
        for i, layer_weights in enumerate(weights):
            new_layer_weights = np.empty_like(layer_weights)
            for j, weight_row in enumerate(layer_weights):
                for k, weight_column in enumerate(weight_row):
                    weights[i][j][k] += delta
                    adjusted_loss = self.compute_loss(training_data)
                    # reset bias
                    weights[i][j][k] -= delta
                    gradient = (adjusted_loss - original_loss) / delta
                    new_layer_weights[j][k] = (
                        weights[i][j][k] - learning_rate * gradient
                    )
            new_weights.append(new_layer_weights)

        self.parameters["biases"] = new_biases
        self.parameters["weights"] = new_weights

    def compute_loss(self, training_data):
        return sum(
            [
                self.loss_function(self.compute_output(training_input), training_output)
                for training_input, training_output in zip(*training_data)
            ]
        )

    def loss_function(self, actual, target):
        return loss_functions[self.hyperparameters["training"]["loss_function"]](
            actual, target
        )

    def activation_function(self, preactivation):
        return activation_functions[self.hyperparameters["activation_function"]](
            preactivation
        )

    def output_activation_function(self, output_preactivation):
        return activation_functions[self.hyperparameters["output_activation_function"]](
            output_preactivation
        )

    def train(self, training_data, test_data):
        pass
