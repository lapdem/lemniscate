import experiment.experiment as ex
import os
import numpy as np

input_folder = "input"
output_folder ="output"
experimetn_output_folder = "network_width_output_"
configuration_file_name = "network_width_experiment.yaml"

lowest_network_width = 5
highest_network_width = 100

networkwidths = [150,200, 250, 300,400,500]#range(lowest_network_width, highest_network_width+lowest_network_width, 5)

for networkwidth in networkwidths:
    experiment = ex.Experiment(input_folder, configuration_file_name, os.path.join(output_folder, experimetn_output_folder + str(int(networkwidth+0.5))))
    #change hidden layer width
    experiment.config["ensemble"]["network"]["hyperparameters"]["layerwidths"][1] = int(networkwidth+0.5)
    result = experiment.run()
    print(result)
