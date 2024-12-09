import json
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "util")))
import colors as colors

plt.rcParams.update({"font.size": 14})

steps_to_plot_for_figures = [0, 100, 500, 1000, 5000, 10000]
steps_plot_for_bias_graph = range(0, 10100, 100)


results_folder = "C:/Users/university/Documents/thesis/results/training_dynamics"

waveform_folders = ["legendre3"]  # , "sin1", "sin2"] #, "cos2"]
result_names = ["Legendre 3"]  # ,  "Sine f=1", "Sine f=2"]# ,  "Cosine f=2"]

biases_by_waveform = []
variances_by_waveform = []

for waveform_folder in waveform_folders:
    with open(os.path.join(results_folder, waveform_folder, "data.json")) as data_file:
        data = json.load(data_file)

    inputs = data["x"]
    outputs = data["y"]
    training_inputs = data["training_x"]
    training_outputs = data["training_y"]

    with open(
        os.path.join(results_folder, waveform_folder, "training_dynamics.json")
    ) as training_dynamics_file:
        training_dynamics = json.load(training_dynamics_file)

    steps = training_dynamics["steps"]
    theoretic_output = np.squeeze(np.array(training_dynamics["theoretic_output"]), -1)
    numeric_outputs = np.squeeze(np.array(training_dynamics["numeric_outputs"]), -1)

    def draw_outputgraphs(steps_to_plot, steps, theoretic_output, numeric_outputs):

        fig, axes = plt.subplots(3, 2, figsize=(9, 10))

        for ax_index, step in enumerate(steps_to_plot):
            ax = axes[ax_index // 2, ax_index % 2]

            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-1.5, 1.5])
            ax.set_xlabel("Network input x")
            ax.set_ylabel("Network output y")

            step_index = steps.index(step)

            for output_index, output in enumerate(numeric_outputs[step_index]):
                ax.plot(
                    inputs,
                    output,
                    "-",
                    color=colors.red_gradient[0],
                    label="Network outputs" if output_index == 0 else "",
                )

            ax.plot(
                inputs,
                theoretic_output[step_index],
                "-",
                color=colors.blue_gradient[2],
                linewidth=3.5,
                label="Theory mean prediction",
            )

            average_output = np.average(numeric_outputs[step_index], axis=0)
            ax.plot(
                inputs,
                average_output,
                "--",
                linewidth=2,
                color="orange",  # colors.bright_colors["bright_yellow"],
                label="Ensemble average",
            )

            ax.plot(
                training_inputs,
                training_outputs,
                "k.",
                label="Training data points",
            )

            ax.legend(loc="lower right", fontsize=10)
            ax.text(0.05, 0.9, f"{step} steps", transform=ax.transAxes, fontsize=16)

        for ax in axes.flat:
            ax.label_outer()

        plt.tight_layout()
        plt.savefig("graphic_v")  # save the figure
        plt.show()
        plt.pause(0.001)
        # save the figure

    draw_outputgraphs(
        steps_to_plot_for_figures, steps, theoretic_output, numeric_outputs
    )

    def calculate_biases_and_variances(
        steps_to_measure, steps, theoretic_output, numeric_outputs
    ):
        biases = []
        variances = []
        mean_squared_errors = []

        for step in steps_to_measure:
            step_index = steps.index(step)
            numeric_outputs_at_step = numeric_outputs[step_index]
            average_output_at_step = np.average(numeric_outputs_at_step, axis=0)
            theoretic_output_at_step = theoretic_output[step_index]

            bias_at_step = np.mean(
                (average_output_at_step - theoretic_output_at_step) ** 2
            )
            variance_at_step = np.mean(
                np.mean((numeric_outputs_at_step - average_output_at_step) ** 2, axis=0)
            )

            biases.append(bias_at_step)
            variances.append(variance_at_step)

            mean_squared_error = np.mean(
                np.mean(
                    (numeric_outputs_at_step - theoretic_output_at_step) ** 2, axis=1
                )
            )
            mean_squared_errors.append(mean_squared_error)

        return biases, variances, mean_squared_errors

    biases, variances, mse = calculate_biases_and_variances(
        steps_to_plot_for_figures, steps, theoretic_output, numeric_outputs
    )

    print(waveform_folder + " Bias % Variance:")
    for bias, variance in zip(biases, variances):
        print(f"{bias:.2e} & {variance:.2e} &")

    biases_full, variances_full, mse_full = calculate_biases_and_variances(
        steps_plot_for_bias_graph, steps, theoretic_output, numeric_outputs
    )

    biases_by_waveform.append(biases_full)
    variances_by_waveform.append(variances_full)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for biases, variances, result_name in zip(
    biases_by_waveform, variances_by_waveform, result_names
):
    # Remove 0 entry
    biases = biases
    variances = variances
    steps_to_plot_trimmed = steps_plot_for_bias_graph

    ax1.plot(steps_to_plot_trimmed, np.sqrt(biases), label=result_name)
    ax2.plot(steps_to_plot_trimmed, variances, label=result_name)

ax1.set_xlabel("Training steps")
ax1.set_ylabel("Mean bias ²")
ax1.legend(fontsize=12)
ax1.set_title("Bias² over time")
ax1.set_xlim(0, 10000)
ax1.set_ylim(0.0, 0.15)
ax2.set_xticks([200] + list(range(2000, 10001, 2000)))

ax2.set_xlabel("Training steps")
ax2.set_ylabel("Mean Variance")
ax2.legend(fontsize=12)
ax2.set_title("Variance over time")
ax2.set_xlim(0, 10000)
ax2.set_ylim(0.0, 0.0015)
# force xtick at 200
ax2.set_xticks([200] + list(range(2000, 10001, 2000)))

plt.tight_layout()
plt.show()
