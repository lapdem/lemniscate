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

        with open(os.path.join(self._input_folder, self._config_file_name)) as config_file:
            self.config = yaml.safe_load(config_file)[self._config_entry_name]
          
        self.data = self.create_data()
        

    def create_data(self):        
        data_config = self.config[self._data_config_entry_name]

        data_source = "source"
        data_source_generate = "generate"
        data_source_from_file = "from_file"

        if(data_config[data_source] == data_source_generate):
            return self._generate_data(data_config[data_source_generate])
        elif((data_config[data_source] == data_source_from_file)):
            return self._load_data_from_file(data_config[data_source_from_file])
        else:
            raise NotImplementedError(f"Data source '{data_config[data_source]}' is not implemented.")

        
    def run(self):

        if not os.path.isdir(self._output_folder):
            os.makedirs(self._output_folder)

        #graphics
        fig = plt.figure()      
        plt.ion()
        plt.show()

        ensemble = Ensemble(self.config[self._ensemble_config_entry_name])
        theory = Theory(self.data, self.config["ensemble"]["network"]["hyperparameters"], 1.0, 1.0)
        
        training_config = self.config["ensemble"]["network"]["hyperparameters"]["training"]
        results_config = self.config["results"]
        steps_per_image = results_config["steps_per_image"]

        for step in range(training_config["max_iterations"]):
            #display data points
            
            #input("Press [enter] to continue.")
            ensemble.evolve(self.data)
            theory.evolve()

            #produce and save graphic every so often
            if step%steps_per_image == 0:
                plt.clf()
                plt.plot(self.data[0], self.data[1].flatten(), "k.", label = "Training data points")          
                
                inputs = np.linspace(0,1,100)
                outputs = ensemble.compute_outputs(inputs)
                for j, output in enumerate(outputs):
                    plt.plot(inputs, np.concatenate(output).ravel(), "r-", label="Ensemble output (numeric)" if j == 0 else "")
                average_output = np.average(outputs, axis=0)
                plt.plot(inputs, np.concatenate(average_output).ravel(),"--", color='orange', linewidth=2, label = "Ensemble output average (numeric)")  
                
                theory_training_outputs = theory.compute_training_ouputs()
                plt.plot(self.data[0], theory_training_outputs, "b+", label="Predicted outputs (analytic)")

                ax = plt.gca()
                ax.set_ylim([-1.5, 1.5])
                ax.legend( loc="lower right")
                plt.text(0.05, 0.9, f"{step:>04} steps", transform = ax.transAxes)
                plt.draw()            
                plt.pause(0.001)
                fig.savefig(os.path.join(self._output_folder, f"image{int(step/steps_per_image):>04}.png"),dpi=fig.dpi)

        #create a video, requires ffmpeg installed    
        #ffmpeg -f image2 -framerate 10 -i output\image%04d.png -vcodec libx264 -crf 15  -pix_fmt yuv420p -y \output\learning.mp4
       
        results = []
        return results

          
    def _generate_data(self, data_generation_config):       
        input_config = data_generation_config["input"]
        input_values = dg.generate_inputs(input_config)

        output_config = data_generation_config["output"]
        output_values = dg.generate_outputs(input_values, output_config)
        
        noise_config = output_config["noise"]
        noise = dg.generate_random_numbers(np.size(output_values), noise_config)        
        output_values += noise

        data = (input_values, output_values)
        return data 


    def _load_data_from_file(self, data_from_file_config):
        raise NotImplementedError