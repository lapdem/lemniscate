import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'util')))
import colors as colors

plt.rcParams.update({'font.size': 14})

results_folder = "C:/Users/university/Documents/thesis/results/training_dynamics/legendre3"
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

# Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]



fig = plt.figure(constrained_layout=True, figsize=(14, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 2])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

ax1.plot(sorted_eigenvalues, marker='o', linestyle='-', color= colors.blue_gradient[0], label="Eigenvalues")
ax1.set_yscale('log')
ax1.set_ylabel("Eigenvalue")

ax1.set_xlabel("Eigenvalue index")
ax1.legend()
ax1.title.set_text("NTK Spectrum")

ax2.plot(training_inputs, sorted_eigenvectors[:, :5], label=[f"Eigenvector with \neigenvalue: {val:.3e}" for i, val in enumerate(sorted_eigenvalues[:5])])
ax2.set_xlabel("Network input x")
ax2.set_ylabel("Network output y")
ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0., labelspacing=2.5, frameon=True)
ax2.set_title("Eigenvectors with largest eigenvalues")

plt.show()


plt.show()
    