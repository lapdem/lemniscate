import experiment.experiment as ex
import os
import numpy as np

input_folder = "input"
output_folder ="output"
experimetn_output_folder = "network_depth_output_"
configuration_file_name = "network_depth_experiment.yaml"

highest_network_depth = 10

for additional_hidden_layers in range(highest_network_depth):
    experiment = ex.Experiment(input_folder, configuration_file_name, os.path.join(output_folder, experimetn_output_folder + str(additional_hidden_layers+1)))
    #change hidden layer width
    for i in range(additional_hidden_layers):
        experiment.config["ensemble"]["network"]["hyperparameters"]["layerwidths"].insert(
            1,
            experiment.config["ensemble"]["network"]["hyperparameters"]["layerwidths"][1]
        )
    result = experiment.run()
    print(result)