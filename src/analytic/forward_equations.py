import math
import numpy as np
import scipy as sp

# analytic solutions for infinite width limit of a neural network during traing
# first order
# see book page 206 following


def calculate_Theta(x, rho, lambda_b, lambda_w, C_b, C_w, n_0, layer):
    # lambda_b, lambda_w, C_b and C_w are currently treated as numbers but should have a layer index
    if layer == 1:
        return lambda_b + lambda_w / n_0 * np.outer(x, x)  # np.einsum("ij,kj", x, x))
    else:
        return (
            lambda_b
            + lambda_w * correlator(rho, calculate_K(x, rho, C_b, C_w, n_0, layer - 1))
            + C_w
            * calculate_Theta(x, rho, lambda_b, lambda_w, C_b, C_w, n_0, layer - 1)
        )


def calculate_K(x, rho, C_b, C_w, n_0, layer):
    # C_b and C_w are currently treated as numbers but should have a layer index
    if layer == 1:
        return C_b + C_w / n_0 * np.outer(x, x)  # np.einsum("ij,kj", x, x))
    else:
        return C_b + C_w * correlator(
            rho, calculate_K(x, rho, C_b, C_w, n_0, layer - 1)
        )


def correlator(rho, K):
    print("calculating correlator for K")
    print(K)
    correlator = np.empty_like(K)
    for delta_1, delta_2 in np.ndindex(K.shape):
        if delta_1 == delta_2:
            correlator[delta_1][delta_2] = sp.integrate.quad(
                lambda phi: rho(phi) ** 2
                * math.exp(-0.5 * phi**2 / K[delta_1][delta_1])
                / math.sqrt(2 * math.pi * K[delta_1][delta_1]),
                -math.inf,
                +math.inf,
            )[0]
        else:

            K_reduced = np.array(
                [
                    [K[delta_1][delta_1], K[delta_1][delta_2]],
                    [K[delta_2][delta_1], K[delta_2][delta_2]],
                ]
            )
            K_reduced_inv = np.linalg.inv(K_reduced)
            K_reduced_det = np.linalg.det(K_reduced)

            def integrand(phi_delta_1, phi_delta_2, K_reduced_inv):
                phi = np.array([phi_delta_1, phi_delta_2])
                return (
                    rho(phi_delta_1)
                    * rho(phi_delta_2)
                    * math.exp(-0.5 * phi.T @ K_reduced_inv @ phi)
                )

            integral = sp.integrate.dblquad(
                integrand,
                -math.inf,
                +math.inf,
                -math.inf,
                +math.inf,
                args=(K_reduced_inv,),
            )[0]

            correlator[delta_1][delta_2] = integral / (
                2 * math.pi * math.sqrt(K_reduced_det)
            )

        print(delta_1, delta_2, correlator[delta_1][delta_2])

    return correlator


def calculate_phi_bar(x, y, t, Theta):
    phi_bar = []
    for alpha, x_alpha in enumerate(x):
        Theta_tilde = np.average(Theta[alpha])
        phi_bar_alpha = (1 - math.exp(-t * Theta_tilde)) * y[alpha]
        phi_bar.append(phi_bar_alpha)
    return phi_bar


# x = [[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]]


# rho = np.tanh
# lambda_b = 0.1
# lambda_w = 0.1
# C_b = 1.0
# C_w = 1.0
# n_0 = 1

# K = calculate_K(x, rho, C_b, C_w, n_0, 3)
# Theta = calculate_Theta(x, rho, lambda_b, lambda_w, C_b, C_w, n_0, 3)
# print(Theta)
