import experiment.experiment as ex
import numpy as np

input_folder = "input"
output_folder = "polynomial_fit_output"
configuration_file_name = "polynomial_fit_experiment.yaml"

highest_order = 7
training_times = []
for order in range(highest_order):
    experiment = ex.Experiment(input_folder, configuration_file_name, output_folder + str(order))
    original_x, original_y = experiment.data
    coefficents = np.polynomial.polynomial.polyfit(np.ndarray.flatten(original_x), np.ndarray.flatten(original_y), order)
    new_y = np.polynomial.polynomial.polyval(original_x, coefficents)
    experiment.data = (original_x, new_y.reshape(-1, 1))
    result = experiment.run()
    print(result)
    training_times.append(result["training_time"])
print(training_times)
