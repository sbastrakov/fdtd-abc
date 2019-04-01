import numpy as np
import scipy.constants


# this value is for sigma E
def _get_sigma_opt(steps, order):
    sigma_opt = np.array([0.0, 0.0, 0.0])
    ## equation (15) in CONVOLUTION PML (CPML): AN EFFICIENT FDTD IMPLEMENTATION OF THE CFS – PML FOR ARBITRARY MEDIA
    ## the same is (17) in  Performance advantages of CPML over UPML absorbing boundary conditions in FDTD algorithm
    for d in range(3):
        sigma_opt[d] = 0.8 * (order + 1) / (scipy.constants.value('characteristic impedance of vacuum') * steps[d])
    return sigma_opt

def get_sigma_max(steps, order, opt_ratio = 1.0):
    return opt_ratio * _get_sigma_opt(steps, order)
