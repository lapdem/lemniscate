import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import util.data_generation as dg
from numeric.ensemble import Ensemble
from analytic.theory import Theory


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

        self.data = self.create_data()

    def create_data(self):
        data_config = self.config[self._data_config_entry_name]

        data_source = "source"
        data_source_generate = "generate"
        data_source_from_file = "from_file"

        if data_config[data_source] == data_source_generate:
            return self._generate_data(data_config[data_source_generate])
        elif data_config[data_source] == data_source_from_file:
            return self._load_data_from_file(data_config[data_source_from_file])
        else:
            raise NotImplementedError(
                f"Data source '{data_config[data_source]}' is not implemented."
            )

    def run(self):

        if not os.path.isdir(self._output_folder):
            os.makedirs(self._output_folder)
       

        ensemble = Ensemble(self.config[self._ensemble_config_entry_name])
        #theory = Theory(
        #    self.data, self.config["ensemble"]["network"]["hyperparameters"], 0.01, 0.01
        #)

        training_config = self.config["ensemble"]["network"]["hyperparameters"]["training"]

        results_config = self.config["results"]
        

        #create subpolts for training
        training_graphic_configs = self.config["graphics"]["training"]
        plt.ion()
        training_fig = plt.figure()
        for training_graphic in training_graphic_configs:
            ax = training_fig.subplots()
            training_graphic["ax"] = ax
                  
            

        training_fig.show()
        
        for step in range(training_config["max_iterations"]):
            
                        
            average_numeric_loss = ensemble.evolve(self.data)
            #theory.evolve()

            #draw training graphs
            for training_graphic in training_graphic_configs:
                if step % training_graphic["steps_per_image"] == 0:
                    ax = training_graphic["ax"]
                    ax.clear()
                    ax.set_xlim(training_graphic["x_lim"])
                    ax.set_ylim(training_graphic["y_lim"])      
                    
                    

                    inputs = np.linspace(0, 1, 100)
                    outputs = ensemble.compute_outputs(inputs)    

                    if "ensemble" in training_graphic["graphs"]:
                        for i, output in enumerate(outputs):
                            ax.plot(
                                inputs,
                                np.concatenate(output).ravel(),
                                "r-",
                                label="Ensemble output (numeric)" if i == 0 else "",
                            )

                    if "ensemble_average" in training_graphic["graphs"]:
                        average_output = np.average(outputs, axis=0)
                        ax.plot(
                            inputs,
                            np.concatenate(average_output).ravel(),
                            "--",
                            color="orange",
                            linewidth=2,
                            label="Ensemble output average (numeric)",
                        )

                    if "data" in training_graphic["graphs"]:
                        ax.plot(
                            self.data[0],
                            self.data[1].flatten(),
                            "k.",
                            label="Training data points",
                        )

                    ax.legend(loc="lower right")
                    ax.text(0.05, 0.9, f"{step:>04} steps", transform=ax.transAxes)

                training_fig.canvas.draw()
                plt.pause(0.001)
                training_fig.savefig(
                    os.path.join(
                        self._output_folder, f"image{int(step/training_graphic["steps_per_image"]):>04}.png"
                    ),
                    dpi=training_fig.dpi,
                )
                
            
            average_numeric_loss_cutoff = training_config.get("average_loss_cutoff", 0.0)
            if average_numeric_loss < average_numeric_loss_cutoff:
                break

        plt.close(training_fig)

        # create a video, requires ffmpeg installed
        # ffmpeg -f image2 -framerate 10 -i output\image%04d.png -vcodec libx264 -crf 15  -pix_fmt yuv420p -y \output\learning.mp4

        results = {
            "training_time": step
        }
        return results

    def _generate_data(self, data_generation_config):
        input_config = data_generation_config["input"]
        input_values = dg.generate_inputs(input_config)

        output_config = data_generation_config["output"]
        output_values = dg.generate_outputs(input_values, output_config)

        noise_config = output_config["noise"]
        #noise = dg.generate_random_numbers(np.size(output_values), noise_config)
        #output_values += noise

        data = (input_values, output_values)
        return data

    def _load_data_from_file(self, data_from_file_config):
        raise NotImplementedError
