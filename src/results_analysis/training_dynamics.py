import json
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'util')))
import colors as colors

plt.rcParams.update({'font.size': 14})

steps_to_plot = [0, 100, 500, 1000, 5000, 10000]

results_folder = "C:/Users/university/Documents/thesis/results/training_dynamics/legendre5"

with open(os.path.join(results_folder, "data.json")) as data_file:
    data = json.load(data_file)

inputs = data["x"]
outputs = data["y"]
training_inputs = data["training_x"]
training_outputs = data["training_y"]

with open(
    os.path.join(results_folder, "training_dynamics.json")
) as training_dynamics_file:
    training_dynamics = json.load(training_dynamics_file)

steps = training_dynamics["steps"]
theoretic_output = np.array(training_dynamics["theoretic_output"])
numeric_outputs = np.array(training_dynamics["numeric_outputs"])


def draw_outputgraphs(steps_to_plot, steps, theoretic_output, numeric_outputs):

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax_index, step in enumerate(steps_to_plot):
        ax = axes[ax_index // 3, ax_index % 3]

        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-1.5, 1.5])
        ax.set_xlabel("Network input x")
        ax.set_ylabel("Network output y")

        step_index = steps.index(step)


        for output_index, output in enumerate(numeric_outputs[step_index]):
            ax.plot(
                inputs,
                np.concatenate(output).ravel(),
                "-",
                color=colors.red_gradient[0],
                label="Network outputs" if output_index == 0 else "",
            )

        ax.plot(
            inputs,
            theoretic_output[step_index],
            "-",
            color=  colors.blue_gradient[1],
            linewidth=4.5,
            label="Theory mean prediction",
        )

        average_output = np.average(numeric_outputs[step_index], axis=0)
        ax.plot(
            inputs,
            np.concatenate(average_output).ravel(),
            "--",
            linewidth=2.5,
            color="orange", #colors.bright_colors["bright_yellow"],
            label="Ensemble average",
        )            

        ax.plot(
            training_inputs,
            training_outputs,
            "k.",
            label="Training data points",
        )

        ax.legend(loc="lower left")
        ax.text(0.05, 0.9, f"{step} steps", transform=ax.transAxes, fontsize=16)

    for ax in axes.flat:
        ax.label_outer()
   

    plt.tight_layout()
    plt.show()
    plt.pause(0.001)

draw_outputgraphs(steps_to_plot, steps, theoretic_output, numeric_outputs)
