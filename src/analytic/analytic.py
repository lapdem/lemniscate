import numpy as np

#analytic solutions for infinite width limit of a neural network during traing
#first order
#see book page 206 following

#C_b bias variance
#C_w weights variance
#n_0 input layer width
#x training data inputs
#rho
#delta training data index
#Theta neural tangent keral




def calculate_K(x, C_b, C_w, n_0, layer):
    #C_b and C_w are currently treated as numbers but should have a layer index
    if(layer == 1):
        #currently only works for x_delta1 being a single number neeeds to be updated for x being a vector
        #the produced matrix will be singular if C_b is 0, need to handle this case in the future
        return (C_b + C_w/n_0 * np.outer(x,x)) #np.einsum("ij,kj", x, x)
    else:
        return (C_b + C_w*correlator(rho1, rho2, calculate_K(x, C_b, C_w, n_0, layer -1)))
    return K_1

def correlator(rho_delta1, rho_delta2, K):

    pass


def calclulate_Theta():
    pass