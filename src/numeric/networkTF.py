from .network import Network
import tensorflow as tf
import numpy as np
import util.data_generation as dg


class NetworkTF(Network):
    def __init__(self, hyper_parameters):
        self.hyperparameters = hyper_parameters

        layerwidths = hyper_parameters["layerwidths"]
        self.model = tf.keras.Sequential()

        # check if GPU is available, Tensorflow will use it by default
        # getting tensorflow to recognise the GPU is a bit of a pain, see https://www.tensorflow.org/guide/gpu
        print(tf.config.list_physical_devices("GPU"))
        print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

        # input layer
        self.model.add(tf.keras.layers.Flatten(input_shape=(layerwidths[0], 1)))

        # hidden layers
        for layer_width in layerwidths[1:-1]:
            self.model.add(
                tf.keras.layers.Dense(
                    layer_width, activation=self.hyperparameters["activation_function"]
                )
            )

        # output layer
        self.model.add(
            tf.keras.layers.Dense(
                layerwidths[-1],
                activation=self.hyperparameters["output_activation_function"],
            )
        )

        print(self.model.summary())

    def initialise_parameters(self, initial_parameter_config, random_number_generator):
        layerwidths = self.hyperparameters["layerwidths"]

        # we could let tensorflow/keras do this but do it ourselves to stay consistent with the plain numpy implementation and theory
        # note that the nparray shapes here are different, biases are just a 1d vector, and weights rows and columns are reversed
        biases = [
            dg.generate_random_numbers(
                (width),
                initial_parameter_config["biases"],
                random_number_generator=random_number_generator,
            )
            for width in layerwidths[1:]
        ]

        weights = [
            self._scale_weights(
                dg.generate_random_numbers(
                    (first_width, second_width),
                    initial_parameter_config["weights"],
                    random_number_generator=random_number_generator,
                ),
                first_width,
            )
            for first_width, second_width in zip(layerwidths[:-1], layerwidths[1:])
        ]

        for layer_index, layer in enumerate(self.model.layers[1:]):
            layer.set_weights([weights[layer_index], biases[layer_index]])

        self.model.compile(
            tf.keras.optimizers.SGD(
                learning_rate=self.hyperparameters["training"]["learning_rate"]
            ),
            loss="mean_squared_error",  # Computes the mean of squares of errors between labels and predictions
        )

    def set_training_data(self, training_data):
        dataset = tf.data.Dataset.from_tensor_slices(
            (training_data[0], training_data[1])
        )
        self.training_batch = dataset.batch(len(training_data[0]))

    def _scale_weights(self, weights, previous_width):
        weight_scaling = self.hyperparameters["scale_weighted_sum"]
        if weight_scaling == "one_over_root_n":
            return np.divide(weights, np.sqrt(previous_width))
        else:
            return weights

    def compute_output(self, input):
        return self.model(input)

    def compute_outputs(self, inputs):
        return self.model.predict(inputs)

    def evolve(self, steps=1):
        # train a single step using the whole data set (non-stochatic gradient descent)
        history = self.model.fit(self.training_batch, epochs=steps)
        if "loss" in history.history:
            return history.history["loss"][-1]
        else:
            loss = self.model.evaluate(self.training_batch, verbose=0)
            return loss
