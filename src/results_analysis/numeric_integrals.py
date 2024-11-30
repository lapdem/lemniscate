import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cubature




def rho(phi):
    return np.tanh(phi)

K_reduced = np.array([[1.0, 1.0], [1.0, 1.00041649]])



K_reduced_inv = sp.linalg.inv(K_reduced)
K_reduced_det = sp.linalg.det(K_reduced)

def qintegrand(phi_delta_1, phi_delta_2):
    phi = np.array([phi_delta_1, phi_delta_2])
    return (
        rho(phi_delta_1)
        * rho(phi_delta_2)
        * math.exp(-0.5 * (phi @ K_reduced_inv @ phi))
    )
    
qresult, qerror  = sp.integrate.nquad(qintegrand, [[-np.inf, np.inf], [-np.inf, np.inf]] )


print(f"2D Correlator Quad Integral Result: {qresult}, Error: {qerror}")


def integrand(phi):
        phi_delta1 = phi[:, 0]
        phi_delta2 = phi[:, 1]

        phi_vectors = phi  # (N, 2) array
        quadratic_form = np.einsum(
            "ij,jk,ik->i", phi_vectors, K_reduced_inv, phi_vectors
        )

        return rho(phi_delta1) * rho(phi_delta2) * np.exp(-0.5 * quadratic_form)

   

bounds = np.array([[-2, 2], [-2, 2]])

result, error = cubature.cubature(
    integrand,  # Function to integrate
    2,  # Input dimension
    1,  # Output dimension
    bounds[:, 0],
    bounds[:, 1],
    vectorized=True,  # False if integrand processes one point at a time
    adaptive="p",
)


print(f"2D Correlator Cub p Integral Result: {result}, Error: {error}")

def transformed_integrand(ab):
    a = ab[:, 0]
    b = ab[:, 1]

    # Compute phi_delta1 and phi_delta2
    phi_delta1 = a / (1 - a**2)
    phi_delta2 = b / (1 - b**2)

    # Compute the Jacobian determinant
    jacobian = (1 + a**2) / (1 - a**2)**2 * (1 + b**2) / (1 - b**2)**2

    # Form the phi vector with transformed variables
    phi_vectors = np.column_stack((phi_delta1, phi_delta2))

    return jacobian * integrand(phi_vectors)

bounds = np.array([[-1, 1], [-1, 1]])

result, error = cubature.cubature(
    integrand,  # Function to integrate
    2,  # Input dimension
    1,  # Output dimension
    bounds[:, 0],
    bounds[:, 1],
    vectorized=True,  # False if integrand processes one point at a time
    adaptive="p",
)
print(f"2D Correlator Cub h Integral Result: {result}, Error: {error}")