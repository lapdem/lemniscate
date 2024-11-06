import secrets
import numpy as np


input_generators = {}
input_generator = lambda f: input_generators.setdefault(f.__name__, f)


@input_generator
def linear(config):
    return np.reshape(np.linspace(config["min"], config["max"], config["size"]),[-1, 1])


@input_generator
def logarithmic(config):
    return np.reshape(np.logspace(config["min"], config["max"], config["size"]),[-1, 1])


@input_generator
def random(config):
    return np.reshape(generate_random_numbers(config["size"], config),[-1, 1])


def generate_inputs(config):
    return input_generators[config["spacing"]](config)


output_generators = {}
output_generator = lambda f: output_generators.setdefault(f.__name__, f)


@output_generator
def legendre(input_values, parameters):
    return np.polynomial.legendre.legval(input_values, parameters["coefficients"])

@output_generator
def sin(input_values, parameters):
    frequency = parameters.get("frequency", 1.0)
    phase = parameters.get("phase", 0.0)
    return np.sin((input_values*2*np.pi*frequency)+phase)


def generate_outputs(input_values, config):
    return output_generators[config["function"]](input_values, config["parameters"])


random_generators = {}
random_generator = lambda f: random_generators.setdefault(f.__name__, f)


@random_generator
def gaussian(
    size,
    config,
    random_number_generator,
):
    return random_number_generator.normal(config["mean"], config["variance"], size)


def generate_random_numbers(size, config, random_number_generator=None):
    if random_number_generator == None:
        random_seed = config.get("random_seed", secrets.randbits(128))
        random_number_generator = np.random.default_rng(random_seed)
    return random_generators[config["distribution"]](
        size, config, random_number_generator
    )
