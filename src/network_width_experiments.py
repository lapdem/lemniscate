import experiment.experiment as ex
import os
import numpy as np

input_folder = "input"
output_folder = "output"
experimetn_output_folder = "network_width_output_"
configuration_file_name = "network_width_experiment.yaml"


networkwidths = [200, 250, 300, 500, 1000, 2000, 2500]

for networkwidth in networkwidths:
    experiment = ex.Experiment(
        input_folder,
        configuration_file_name,
        os.path.join(output_folder, experimetn_output_folder + str(networkwidth)),
    )
    # change hidden layer width
    experiment.config["ensemble"]["network"]["hyperparameters"]["layerwidths"][
        1
    ] = networkwidth
    result = experiment.run()
    print(result)
