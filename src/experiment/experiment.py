import yaml
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import util.data_generation as dg
from numeric.ensemble import Ensemble
from analytic.theory import Theory
import util.colors as colors
import tensorflow as tf


class Experiment:
    _config_entry_name = "experiment"
    _data_config_entry_name = "data"
    _ensemble_config_entry_name = "ensemble"

    def __init__(self, input_folder, config_file_name, output_folder) -> None:
        self._input_folder = input_folder
        self._config_file_name = config_file_name
        self._output_folder = output_folder

        with open(
            os.path.join(self._input_folder, self._config_file_name)
        ) as config_file:
            self.config = yaml.safe_load(config_file)[self._config_entry_name]

        self.data, self.training_data = self.create_data()

    def create_data(self):
        data_config = self.config[self._data_config_entry_name]

        data_source = "source"
        data_source_generate = "generate"
        data_source_from_file = "from_file"
        data_source_mnist = "mnist"

        if data_config[data_source] == data_source_generate:
            return self._generate_data(data_config[data_source_generate])
        elif data_config[data_source] == data_source_from_file:
            return self._load_data_from_file(data_config[data_source_from_file])
        elif data_config[data_source] == data_source_mnist:
            return self._load_mnist(data_config[data_source_mnist])
        else:
            raise NotImplementedError(
                f"Data source '{data_config[data_source]}' is not implemented."
            )

    def run(self):

        if not os.path.isdir(self._output_folder):
            os.makedirs(self._output_folder)

        # write config to output folder
        with open(os.path.join(self._output_folder, "experiment.yaml"), "w") as outfile:
            yaml.dump({"experiment": self.config}, outfile)

        with open(os.path.join(self._output_folder, "data.json"), "w") as outfile:
            data = {
                "x": self.data[0].tolist(),
                "y": self.data[1].tolist(),
                "training_x": self.training_data[0].tolist(),
                "training_y": self.training_data[1].tolist(),
            }
            json.dump(data, outfile)

        ensemble = Ensemble(self.config[self._ensemble_config_entry_name])
        theory = Theory(self.config["ensemble"]["network"]["hyperparameters"])

        training_dynamics = {
            "steps": [],
            "numeric_losses": [],
            "numeric_outputs": [],
            "theoretic_loss": [],
            "theoretic_output": [],
        }

        theory.initialise_parameters(
            self.config["ensemble"]["network"]["initial_parameters"]["generate"]
        )

        ensemble.set_training_data(self.training_data)

        theory.set_data(self.data)
        theory.set_training_data(self.training_data)

        # create subplpts for training
        training_graphic_configs = self.config["graphics"]["training"]
        plt.ion()
        training_fig = plt.figure()
        for training_graphic in training_graphic_configs:
            ax = training_fig.add_subplot()
            training_graphic["ax"] = ax

        training_fig.show()

        results_config = self.config["results"]

        numeric_training_times = [
            0 for _ in range(self.config["ensemble"]["number_of_networks"])
        ]
        theoretical_training_time = 0

        step_range = []
        step_config = self.config["steps"]
        if step_config["source"] == "range":
            step_range = range(
                step_config["range"]["start"],
                step_config["range"]["stop"],
                step_config["range"]["step"],
            )
        if step_config["source"] == "list":
            step_range = step_config["list"]

        previous_step = 0
        for step_index, step in enumerate(step_range):

            theoretic_loss = theory.evolve(step - previous_step)

            numeric_losses = ensemble.evolve(step - previous_step)

            previous_step = step
            print(f"Step {step} completed")
            training_dynamics["steps"].append(step)
            training_dynamics["numeric_losses"].append(numeric_losses)
            training_dynamics["theoretic_loss"].append(theoretic_loss)
            print(f"theoretic loss {theoretic_loss}")
            print(f"numeric losses {numeric_losses}")

            inputs = np.linspace(0, 1, 100)
            reshaped_inputs = np.reshape(inputs, [-1, 1, 1])
            outputs = ensemble.compute_outputs(reshaped_inputs)
            training_dynamics["numeric_outputs"].append(
                [output.tolist() for output in outputs]
            )

            theory_output = theory.compute_output()
            training_dynamics["theoretic_output"].append(
                [theory_output_entry.tolist() for theory_output_entry in theory_output]
            )
            # draw training graphs
            for training_graphic in training_graphic_configs:
                ax = training_graphic["ax"]
                ax.clear()
                ax.set_xlim(training_graphic["x_lim"])
                ax.set_ylim(training_graphic["y_lim"])

                if "ensemble" in training_graphic["graphs"]:
                    for i, output in enumerate(outputs):
                        ax.plot(
                            inputs,
                            np.concatenate(output).ravel(),
                            "-",
                            color=colors.red_gradient[0],
                            label="Network outputs" if i == 0 else "",
                        )

                if "training_data" in training_graphic["graphs"]:
                    ax.plot(
                        self.training_data[0],
                        self.training_data[1].flatten(),
                        "k.",
                        label="Training data points",
                    )

                if "theory" in training_graphic["graphs"]:

                    ax.plot(
                        self.data[0],
                        theory_output,
                        "-",
                        color=colors.blue_gradient[2],
                        linewidth=3,
                        label="Theory mean prediction",
                    )

                if "ensemble_average" in training_graphic["graphs"]:
                    average_output = np.average(outputs, axis=0)
                    ax.plot(
                        inputs,
                        np.concatenate(average_output).ravel(),
                        "--",
                        color=colors.bright_colors["bright_yellow"],
                        label="Ensemble average",
                    )

                ax.legend(loc="lower right")
                ax.text(0.05, 0.9, f"{step} steps", transform=ax.transAxes)

                training_fig.canvas.draw()
                plt.pause(0.001)
                training_fig.savefig(
                    os.path.join(
                        self._output_folder,
                        f"image{step_index:>04}.png",
                    ),
                    dpi=training_fig.dpi,
                )

            for i, numeric_loss in enumerate(numeric_losses):
                if (
                    numeric_loss < results_config["loss_threshold"]
                    and numeric_training_times[i] == 0
                ):
                    numeric_training_times[i] = step

            if (
                theoretic_loss < results_config["loss_threshold"]
                and theoretical_training_time == 0
            ):
                theoretical_training_time = step

            if (
                all(
                    [
                        numeric_training_time > 0
                        for numeric_training_time in numeric_training_times
                    ]
                )
                and theoretical_training_time > 0
            ):
                break

        plt.close(training_fig)

        results = {
            "numeric_training_times": numeric_training_times,
            "theoretic_training_time": theoretical_training_time,
            "theoretic_ntk": theory.get_NTK().tolist(),
        }

        with open(os.path.join(self._output_folder, "results.json"), "w") as outfile:
            json.dump(results, outfile)

        with open(
            os.path.join(self._output_folder, "training_dynamics.json"), "w"
        ) as outfile:
            json.dump(training_dynamics, outfile)

        return results

    def _generate_data(self, data_generation_config):
        input_config = data_generation_config["input"]
        input_values = dg.generate_inputs(input_config)

        output_config = data_generation_config["output"]
        output_values = dg.generate_outputs(input_values, output_config)

        # noise_config = output_config["noise"]
        # noise = dg.generate_random_numbers(np.size(output_values), noise_config)
        # output_values += noise

        data = (input_values, output_values)
        training_data = self._generate_training_data(data, data_generation_config)
        return (data, training_data)

    def _generate_training_data(self, data, data_generation_config):
        training_data_ratio = data_generation_config["input"].get(
            "training_data_ratio", 0.3
        )
        data_size = len(data[0])
        training_data_size = int(data_size * training_data_ratio)
        indices = np.linspace(0, data_size - 1, training_data_size, dtype=int)
        training_inputs = data[0][indices]
        training_outputs = data[1][indices]
        return training_inputs, training_outputs

    def _load_data_from_file(self, data_from_file_config):
        raise NotImplementedError

    def _load_mnist(self, data_from_mnist_config):
        (x_train, y_train), (x_text, y_test) = tf.keras.datasets.mnist.load_data()


# to create video use command
# ffmpeg -f image2 -framerate 10 -i C:\Users\university\Documents\thesis\lemniscate\output\image%04d.png -vcodec libx264 -crf 15  -pix_fmt yuv420p -y C:\Users\university\Documents\thesis\lemniscate\output\learning.mp4
