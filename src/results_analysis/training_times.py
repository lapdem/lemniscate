import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "util")))
import colors as colors
#plt.rcParams.update({'font.size': 14})

results_folder = "C:/Users/university/Documents/thesis/results/training_times"

highest_degree = 4

results = []

for degree in range(highest_degree + 1):
    with open(
        os.path.join(results_folder, f"legendre{degree}", "results.json")
    ) as results_file:
        results.append(json.load(results_file))


with open(os.path.join(results_folder, "legendre0", "data.json")) as data_file:
    data = json.load(data_file)

training_inputs = np.array(data["training_x"])
inputs = np.array(data["x"])

training_indices = np.where(np.isin(inputs, training_inputs))[0]

ntk = np.array(results[0]["theoretic_ntk"])
training_ntk = ntk[np.ix_(training_indices, training_indices)]

eigenvalues, eigenvectors = np.linalg.eigh(training_ntk)
# sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

numeric_training_time_mean = []
numeric_training_time_std = []
theoretic_training_time = []

for result in results:
    numeric_training_times = np.array(result["numeric_training_times"])
    numeric_training_time_mean.append(np.mean(numeric_training_times))
    numeric_training_time_std.append(np.std(numeric_training_times, ddof=1))
    theoretic_training_time.append(result["theoretic_training_time"])


plt.plot(
    sorted_eigenvalues[: highest_degree + 1],
    theoretic_training_time,
    label="Theory prediction",
    linestyle="--",
    linewidth=2,
    color=colors.blue_gradient[0],
)

plt.errorbar(
    sorted_eigenvalues[: highest_degree + 1],
    numeric_training_time_mean,
    yerr=numeric_training_time_std,
    label="Network training times",
    color=colors.red_gradient[0],
    linewidth=3,
    fmt="none",
    capsize=5,  # Add caps to the error bars
)

for i in range(highest_degree + 1):
    if i == 0:
        plt.annotate(
            f"Legendre {i}",
            (sorted_eigenvalues[i], numeric_training_time_mean[i]),
            textcoords="offset points",
            xytext=(-15, 25),
            ha="center",
        )
    elif i == highest_degree:
        plt.annotate(
            f"Legendre {i}",
            (sorted_eigenvalues[i], numeric_training_time_mean[i]),
            textcoords="offset points",
            xytext=(15, -25),
            ha="center",
        )
    
    else:
        plt.annotate(
            f"Legendre {i}",
            (sorted_eigenvalues[i], numeric_training_time_mean[i]),
            textcoords="offset points",
            xytext=(-10, -5),
            ha="right",
        )

plt.xlabel("Eigenvalues")
plt.ylabel("Training steps")
plt.xscale("log")
plt.yscale("log")
plt.title("Training time in relation to eigenvalues")
plt.legend()
plt.show()
