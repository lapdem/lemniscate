#TODO: update this so it again correctly implements the Network class

import numpy as np
import util.data_generation as dg
from util.activation_functions import activation_functions
from .network import Network

loss_functions = {}
loss_function = lambda f: loss_functions.setdefault(f.__name__, f)

@loss_function
def mean_squared_error(actual, target):
    return (np.square(actual - target)).mean()

class NetworkNP(Network):

    def __init__(self, hyper_parameters):
        self.hyperparameters = hyper_parameters
        self.activation_function = activation_functions[self.hyperparameters["activation_function"]]
        self.output_activation_function = activation_functions[self.hyperparameters["output_activation_function"]]

    def _compute_activation(self, preactivation):
        return self.activation_function(preactivation)

    def initialise_parameters(self, initial_parameter_config, random_number_generator):
        self.parameters = {}
        layerwidths = self.hyperparameters["layerwidths"]
        biases_sizes = layerwidths[1:]
        weights_sizes = list(zip(layerwidths[:-1], layerwidths[1:]))
        self.parameters["biases"] = [
            dg.generate_random_numbers(
                (width, 1),
                initial_parameter_config["biases"],
                random_number_generator=random_number_generator,
            )
            for width in layerwidths[1:]
        ]
        self.parameters["weights"] = [
            dg.generate_random_numbers(
                (second_width, first_width),
                initial_parameter_config["weights"],
                random_number_generator=random_number_generator,
            )
            for first_width, second_width in zip(layerwidths[:-1], layerwidths[1:])
        ]

    def compute_activations(self, input):        
        layer_activation = np.reshape(input, (-1,1))
        activations = [layer_activation]
        preactivations = []
        for i, (layer_biases, layer_weights) in enumerate(
            zip(self.parameters["biases"], self.parameters["weights"])
        ):
            layer_preactivation = (
                np.dot(layer_weights, layer_activation) / np.sqrt(layer_weights.shape[0]) + layer_biases
            )
            preactivations.append(layer_preactivation)
            if i < (len(self.parameters["biases"]) - 1):
                layer_activation = self.activation_function(layer_preactivation)
            else:
                layer_activation = self.output_activation_function(layer_preactivation)
            activations.append(layer_activation)
            
        return (preactivations, activations)
    
    def compute_output(self, input):    
        return self.compute_activations(input)[1][-1]
    
    def back_propagate(self, training_data):
        training_config = self.hyperparameters["training"]
        learning_rate = training_config["learning_rate"]

        bias_gradients = [np.zeros(b.shape) for b in self.parameters["biases"]]
        weight_gradients = [np.zeros(w.shape) for w in self.parameters["weights"]]

        for training_input, training_output in zip(*training_data):
            sample_bias_gradients = [np.zeros(b.shape) for b in self.parameters["biases"]]
            sample_weight_gradients = [np.zeros(w.shape) for w in self.parameters["weights"]]

            #feed forward
            preactivations, activations = self.compute_activations(training_input)

            #backward pass
            gradient = (activations[-1]- training_output) * self.output_activation_function.derivative(preactivations[-1]) 
            sample_bias_gradients[-1] = gradient
            sample_weight_gradients[-1] = np.dot(gradient, activations[-2].transpose())

            for layers_from_back in range(2, len(self.hyperparameters["layerwidths"])):
                layer_preactivation = preactivations[-layers_from_back]                
                gradient = np.dot(self.parameters["weights"][-layers_from_back+1].transpose(), gradient) * self.activation_function.derivative(layer_preactivation)
                sample_bias_gradients[-layers_from_back] = gradient
                sample_weight_gradients[-layers_from_back] = np.dot(gradient, activations[-layers_from_back-1].transpose())

            bias_gradients = [bias_gradient+sample_bias_gradient for bias_gradient, sample_bias_gradient in zip(bias_gradients, sample_bias_gradients)]
            weight_gradients = [weight_gradient+sample_weight_gradient for weight_gradient, sample_weight_gradient in zip(weight_gradients, sample_weight_gradients)]
        
        self.parameters["biases"] = [bias-learning_rate*bias_gradient
                    for bias, bias_gradient in zip(self.parameters["biases"], bias_gradients)]   
        self.parameters["weights"] = [weight-learning_rate*weight_gradient
                    for weight, weight_gradient in zip(self.parameters["weights"], weight_gradients)]
        
        return self.compute_loss(training_data)
        



    def gradient_descend(self, training_data):
        training_config = self.hyperparameters["training"]

        learning_rate = training_config["learning_rate"]
        delta = training_config["numeric_delta"]

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
                    # reset weights
                    weights[i][j][k] -= delta
                    gradient = (adjusted_loss - original_loss) / delta
                    new_layer_weights[j][k] = (
                        weights[i][j][k] - learning_rate * gradient
                    )
            new_weights.append(new_layer_weights)

        self.parameters["biases"] = new_biases
        self.parameters["weights"] = new_weights

        return self.compute_loss(training_data)
    
    def evolve(self, training_data):
        training_config = self.hyperparameters["training"]

        if(training_config["method"] == "backprop"):
            return self.back_propagate(training_data)
        else:
            return self.gradient_descend(training_data)
            

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

   

