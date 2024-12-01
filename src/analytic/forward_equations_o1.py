import math
import numpy as np
import scipy as sp
import cubature
from numba import jit

import util.matrix_utilities as matrix_utilities
from util.caching import cached


# analytic solutions for infinite width limit of a neural network during traing
# first order
# see book page 206 following


def calculate_Theta(x, rho, lambda_b, lambda_w, C_b, C_w, n_0, layer):
    # lambda_b, lambda_w, C_b and C_w are currently treated as numbers but should have a layer index
    if layer == 1:
        Theta = lambda_b + lambda_w / n_0 * np.einsum("ij,kj", x, x)
    else:
        Theta = (
            lambda_b
            + (
                lambda_w
                * correlator(rho, calculate_K(x, rho, C_b, C_w, n_0, layer - 1))
            )
            + (
                C_w
                * calculate_Theta(x, rho, lambda_b, lambda_w, C_b, C_w, n_0, layer - 1)
                * correlator(
                    rho.derivative, calculate_K(x, rho, C_b, C_w, n_0, layer - 1)
                )
            )
        )

    # Theta = matrix_utilities.nearestPDh(Theta)
    print("Theta")
    print("Eigenvalues of Theta:", sp.linalg.eig(Theta)[0])
    return Theta


@cached
def calculate_K(x, rho, C_b, C_w, n_0, layer):
    # C_b and C_w are currently treated as numbers but should have a layer index
    if layer == 1:
        K = C_b + C_w / n_0 * np.einsum("ij,kj", x, x)
    else:
        K = C_b + C_w * correlator(rho, calculate_K(x, rho, C_b, C_w, n_0, layer - 1))

    # K = matrix_utilities.nearestPDh(K)
    return K


@cached
def correlator(rho, K):
    print("Calculating Correlator...")
    print("Eigenvalues of K:", sp.linalg.eigh(K)[0])
    # numeric_stability_factor = 0.00
    K = K  # + (numeric_stability_factor * np.eye(K.shape[0]) * np.trace(K) / K.shape[0])
    print("Regularised K matrix:")
    print("Eigenvalues of regularised K:", sp.linalg.eigh(K)[0])
    correlator = np.empty_like(K)
    for delta_1 in range(K.shape[0]):
        for delta_2 in range(delta_1, K.shape[1]):
            if delta_1 == delta_2:
                K_entry = K[delta_1][delta_1]
                correlator[delta_1][delta_2] = correlator_1D(rho, K_entry)
            else:
                K_reduced = np.array(
                    [
                        [K[delta_1][delta_1], K[delta_1][delta_2]],
                        [K[delta_2][delta_1], K[delta_2][delta_2]],
                    ]
                )
                correlator[delta_1][delta_2] = correlator_2D(rho, K_reduced)
                correlator[delta_2][delta_1] = correlator[delta_1][
                    delta_2
                ]  # Symmetric assignment
            print(
                f"Correlator value at ({delta_1}, {delta_2}): {correlator[delta_1][delta_2]}"
            )

    # print("Eigenvalues of correlator before regularisation:")
    # the correlator better be positive definite
    # inaccuracies in the numeric integration can lead to negative eigenvalues, so we regularise the correlator
    # correlator = matrix_utilities.nearestPDh(correlator)

    print("Eigenvalues of correlator:", sp.linalg.eigh(correlator)[0])
    return correlator


def correlator_1D(rho, K_entry, numeric_stability_factor=0.00):
    K_entry = K_entry  # + numeric_stability_factor

    # using einsum and vectorization seems to be quicker
    # than just in time compilation which does not support einsum
    def integrand(phi):
        # Ensure phi is a 2D array, even if a 1D input is passed
        phi = np.atleast_2d(phi)
        # Evaluate rho and compute the result for each point
        return np.einsum("i,i->i", rho(phi[:, 0]), rho(phi[:, 0])) * np.exp(
            -0.5 * phi[:, 0] ** 2 / K_entry
        )

    bounds = np.array([[-5, 5]])

    result, error = cubature.cubature(
        integrand,  # Function to integrate
        1,  # Input dimension
        1,  # Output dimension
        bounds[:, 0],
        bounds[:, 1],
        vectorized=True,  # False if integrand processes one point at a time
        adaptive="p",
    )

    print(f"1D Correlator Integral Result: {result}, Error: {error}")

    return result[0] / math.sqrt(2 * math.pi * K_entry)


def correlator_2D(rho, K_reduced):
    K_reduced_inv = sp.linalg.inv(K_reduced)
    K_reduced_det = sp.linalg.det(K_reduced)

    # using einsum and vectorization seems to be quicker
    # than just in time compilation which does not support einsum
    def integrand(phi):
        phi_delta1 = phi[:, 0]
        phi_delta2 = phi[:, 1]

        phi_vectors = phi  # (N, 2) array
        quadratic_form = np.einsum(
            "ij,jk,ik->i", phi_vectors, K_reduced_inv, phi_vectors
        )

        return rho(phi_delta1) * rho(phi_delta2) * np.exp(-0.5 * quadratic_form)

    bounds = np.array([[-5, 5], [-5, 5]])

    result, error = cubature.cubature(
        integrand,  # Function to integrate
        2,  # Input dimension
        1,  # Output dimension
        bounds[:, 0],
        bounds[:, 1],
        vectorized=True,  # False if integrand processes one point at a time
        adaptive="p",
    )

    print(f"2D Correlator Integral Result: {result}, Error: {error}")

    return result / (2 * math.pi * math.sqrt(K_reduced_det))


def calculate_phi_bar(x, y, t, Theta, training_indices, output_indices):
    phi_bar = []
    Theta_tilde = Theta[np.ix_(training_indices, training_indices)]
    Theta_tilde_inverse = np.linalg.inv(
        Theta_tilde
    )  # + numeric_stability_factor * np.eye(Theta_tilde.shape[0])) #* np.abs(np.trace(Theta_tilde))
    Theta_delta_alpha = Theta[np.ix_(output_indices, training_indices)]
    y_alpha = y[np.ix_(training_indices)]
    phi_bar = (
        (Theta_delta_alpha @ Theta_tilde_inverse)
        @ (np.eye(Theta_tilde.shape[0]) - sp.linalg.expm(-Theta_tilde * t))
    ) @ y_alpha
    return phi_bar
