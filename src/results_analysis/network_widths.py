import os
import json
import matplotlib.pyplot as plt
import numpy as np


results_folder ="C:/Users/university/Documents/thesis/results/width_experiments"
theory_output_folder = "network_width_output_5"
ensemble_output_folder = "network_width_output_"

lowest_network_width = 5

highest_network_width = 100

networkwidths =   range(lowest_network_width, highest_network_width+lowest_network_width, 5)


with open(os.path.join(results_folder, theory_output_folder, "training_dynamics.json")) as training_dynamics_file:
    training_dynamics = json.load(training_dynamics_file)
    steps = training_dynamics["steps"]
    theoretic_output = training_dynamics["theoretic_output"]

numeric_outputs = []
for networkwidth in range(lowest_network_width, highest_network_width+lowest_network_width, 5):
    with open(os.path.join(results_folder, ensemble_output_folder + str(networkwidth), "training_dynamics.json")) as training_dynamics_file:
        training_dynamics = json.load(training_dynamics_file)
        numeric_outputs.append(training_dynamics["numeric_outputs"])


for step_index, step in enumerate(steps):
    numeric_outputs_at_step_raw = [numeric_outputs[width_index][step_index] for width_index in range(len(networkwidths))]
    numeric_outputs_at_step = np.squeeze(np.array(numeric_outputs_at_step_raw), -1)
    theoretic_output_at_step = np.squeeze(np.array(theoretic_output[step_index]), -1)
    
    biases = []
    variances = []
    mean_squared_errors = []
    for numeric_outputs_at_width in numeric_outputs_at_step:
        numeric_average = np.mean(numeric_outputs_at_width, axis=0)
        
        bias = np.mean((numeric_average - theoretic_output_at_step) ** 2)
        variance = np.mean(
            np.mean((numeric_outputs_at_width - numeric_average) ** 2, axis=0)
        )

        biases.append(bias)
        variances.append(variance)

        mean_squared_error= np.mean(np.mean((numeric_outputs_at_width - theoretic_output_at_step) ** 2, axis=1))
        mean_squared_errors.append(mean_squared_error)


    
    
    nw = np.array(networkwidths)[5:]    
    biases = np.sqrt(biases)[5:]
    variances = np.array(variances)[5:]
    error_on_bias = 2*np.sqrt(biases)*np.sqrt(variances/30)
    plt.errorbar(1.0/nw, biases, yerr=error_on_bias, label="Bias", color = "red", marker='o')
    #plt.plot(1.0/nw, biases, label="Bias", color = "red", marker='o')
    poly_fit = np.polynomial.polynomial.Polynomial.fit(1/nw, biases, w=error_on_bias, deg=[1], domain=[0,1], window=[0,1])
    print(poly_fit.coef)
    plt.plot(1/nw, poly_fit(1/nw), label="Linear fit", color = "red")
    #plt.plot(networkwidths, variances, label="Variance", color = "green")
    #plt.plot(networkwidths, mean_squared_errors, label="Mean squared error", color = "blue")    
    plt.xlabel("Network width")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
    plt.pause(0.001)    
            

    
    #linear fit mean squared error proportional to 1/n
    #poly_fit = np.polynomial.polynomial.Polynomial.fit(1/np.array(networkwidths), mean_squared_errors, deg=[1], domain=[0,1], window=[0,1])
    #plt.plot(1/np.array(networkwidths), poly_fit(1/np.array(networkwidths)), label="Linear fit", color = "blue")


    
#do the same for networks at various depths
results_folder ="C:/Users/university/Documents/thesis/results/depth_experiments"
output_folder = "network_depth_output_"
highest_network_depth = 8

biases_by_depth = []
variances_by_depth = []

network_depths = range(1, highest_network_depth+1)

for network_depth in network_depths:
    with open(os.path.join(results_folder, output_folder + str(network_depth), "training_dynamics.json")) as training_dynamics_file:
        training_dynamics = json.load(training_dynamics_file)
        numeric_outputs = training_dynamics["numeric_outputs"]
        theoretic_output = training_dynamics["theoretic_output"]
        steps = training_dynamics["steps"]
        
        biases = []
        variances = []
        mean_squared_errors = []
        for step_index, step in enumerate(steps):
            numeric_outputs_at_step = np.squeeze(np.array(numeric_outputs[step_index]), -1)
            theoretic_output_at_step = np.squeeze(np.array(theoretic_output[step_index]), -1)
            
            numeric_average = np.mean(numeric_outputs_at_step, axis=0)
            
            bias = np.mean((numeric_average - theoretic_output_at_step) ** 2)
            variance = np.mean(
                np.mean((numeric_outputs_at_step - numeric_average) ** 2, axis=0)
            )

            biases.append(bias)
            variances.append(variance)

            mean_squared_error= np.mean(np.mean((numeric_outputs_at_step - theoretic_output_at_step) ** 2, axis=1))
            mean_squared_errors.append(mean_squared_error)
            
        biases_by_depth.append(biases)
        variances_by_depth.append(variances)


for step_index, step in enumerate(steps):
    biases_at_step = np.sqrt([biases[step_index] for biases in biases_by_depth])
    variances_at_step = [variances[step_index] for variances in variances_by_depth]
    plt.plot(network_depths, biases_at_step, label="Bias")
    poly_fit = np.polynomial.polynomial.Polynomial.fit(network_depths[:3], biases_at_step[:3], deg=[1], domain=[1,3], window=[1,3])
    print(poly_fit.coef)
    #plt.plot(range(highest_network_depth), variances_at_step, label="Variance")
    plt.xlabel("Network depth")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
    plt.pause(0.001)

    