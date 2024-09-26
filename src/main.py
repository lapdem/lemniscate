import experiment.experiment as ex

input_folder = "input"
output_folder = "output"
configuration_file_name = "experiment.yaml"

experiment = ex.Experiment(input_folder, configuration_file_name, output_folder)
experiment.run()
   
a = 1
