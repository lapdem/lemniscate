import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "util")))
import colors as colors
from scipy.optimize import curve_fit


results_folder = "C:/Users/university/Documents/thesis/results/width_and_depth"

width_folder = "network_width_output_"

network_widths = [200, 250, 300, 500, 1000, 2000, 2500]
width_results = []

for network_width in network_widths:
    with open(
        os.path.join(results_folder, width_folder + str(network_width), "results.json")
    ) as results_file:
        width_results.append(json.load(results_file))

numeric_training_time_mean_by_width = []
numeric_training_time_std_by_width = []
theoretic_training_time_by_width = []
numeric_inverse_mean = []
for result in width_results:
    numeric_training_times = np.array(result["numeric_training_times"])
    numeric_training_time_mean_by_width.append(np.mean(numeric_training_times))
    numeric_training_time_std_by_width.append(np.std(numeric_training_times, ddof=1))
    theoretic_training_time_by_width.append(result["theoretic_training_time"])

depth_folder = "network_depth_output_"
network_depths = []
depth_results = []

for network_depth in network_depths:
    with open(
        os.path.join(results_folder, depth_folder + str(network_depth), "results.json")
    ) as results_file:
        depth_results.append(json.load(results_file))

numeric_training_time_mean_by_depth = []
numeric_training_time_std_by_depth = []
theoretic_training_time_by_depth = []
for result in depth_results:
    numeric_training_times = np.array(result["numeric_training_times"])
    numeric_training_time_mean_by_depth.append(np.mean(numeric_training_times))
    numeric_training_time_std_by_depth.append(np.std(numeric_training_times, ddof=1))
    theoretic_training_time_by_depth.append(result["theoretic_training_time"])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ensure all our variables are numpy arrays
network_widths = np.array(network_widths)
theoretic_training_time_by_width = np.array(theoretic_training_time_by_width)
numeric_training_time_mean_by_width = np.array(numeric_training_time_mean_by_width)
numeric_training_time_std_by_width = np.array(numeric_training_time_std_by_width)
theoretic_training_time_by_width = np.array(theoretic_training_time_by_width)
# theory stays constant
ax1.plot(
    [0, 3000],
    theoretic_training_time_by_width[:2],
    label="Theory prediction",
    linestyle="-",
    linewidth=2,
    color=colors.blue_gradient[0],
)

ax1.errorbar(
    network_widths,
    numeric_training_time_mean_by_width,
    yerr=numeric_training_time_std_by_width,
    label="Numeric training time",
    linestyle="--",
    linewidth=2,
    color=colors.red_gradient[0],
    elinewidth=2,  # Make caps wider
    capsize=5,  # Add caps to the error bars
    capthick=2,
)

ax1.set_xlabel("Network width")
ax1.set_ylabel("Training time")
# do not show error bars in legend
ax1.legend(loc="lower right")
ax1.title.set_text("Training time vs network width")
# force xtick at 200
ax1.set_xticks(ax1.get_xticks().tolist() + [200])
ax1.set_xlim(0, 2600)


ntk_correction = (
    abs(theoretic_training_time_by_width - numeric_training_time_mean_by_width)
    / numeric_training_time_mean_by_width
)
error_nom = numeric_training_time_std_by_width / abs(
    theoretic_training_time_by_width - numeric_training_time_mean_by_width
)
error_denom = numeric_training_time_std_by_width / numeric_training_time_mean_by_width
ntk_correction_error = ntk_correction * (error_nom + error_denom)
ax2.errorbar(
    1 / network_widths,
    ntk_correction,
    yerr=ntk_correction_error,
    label="NTK correction scale",
    fmt="none",
    color=colors.red_gradient[0],
    elinewidth=2,  # Make caps wider
    capsize=5,  # Add caps to the error bars
    capthick=2,
)


# add caps to error bars
# Define the model function
def linear(x, m):
    return m * x


popt, pcov = curve_fit(
    linear,
    1 / network_widths,
    ntk_correction,
    sigma=ntk_correction_error,
    absolute_sigma=True,
)
m_fit = popt[0]
print(f"Fitted parameter: m = {m_fit:.3f}")

fit = m_fit * 1 / network_widths
# do a pure first order fit
# plot the fit
r_squared = 1 - np.sum((ntk_correction - fit) ** 2) / np.sum(
    (ntk_correction - np.mean(ntk_correction)) ** 2
)
ax2.plot(
    [0, 0.06],
    [0, 0.06 * m_fit],
    label=f"Linear Fit, R²={r_squared:.2f}",
    color=colors.blue_gradient[2],
    linestyle="--",
    linewidth=2,
)
# calculate chi squared
chi_squared = np.sum((ntk_correction - fit) ** 2 / ntk_correction_error**2)
print(chi_squared)
# calculate r^2


ax1.yaxis.set_label_position("left")
ax2.yaxis.set_label_position("right")
ax2.set_xlabel("1/Network width")
ax2.set_ylabel("NTK correction scaling")
ax2.title.set_text("NTK correction scaling")
ax2.legend(loc="lower right")
ax2.set_xlim(0, 0.006)
ax2.set_ylim(-0.05, 1)
fig.tight_layout()
plt.show()
