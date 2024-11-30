
import numpy as np
import json
import os

results_folder = "C:/Users/university/Documents/thesis/results/training_dynamics/legendre5"
with open(os.path.join(results_folder, "results.json")) as results_file:
	results = json.load(results_file)

with open(os.path.join(results_folder, "data.json")) as data_file:
	data = json.load(data_file)

training_inputs = np.array(data["training_x"])
inputs= np.array(data["x"])

training_indices = np.where(np.isin(inputs, training_inputs))[0]


ntk = np.array(results["theoretic_ntk"])

# Extract the submatrix of `ntk` corresponding to the training inputs using the training indices
training_ntk = ntk[np.ix_(training_indices, training_indices)]


eigenvalues, eigenvectors = np.linalg.eigh(training_ntk)

training_ntk_condition_number = np.linalg.cond(training_ntk)


training_ntk_perturbed  = np.linalg.inv(np.linalg.inv(training_ntk))
#


eigenvalues_perturbed, eigenvectors_perturbed = np.linalg.eigh(training_ntk_perturbed)

diff = (eigenvalues-eigenvalues_perturbed)/eigenvalues
print(diff)
