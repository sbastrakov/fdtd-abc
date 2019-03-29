import numpy as np


# A set of coefficients for E or B field
class Coefficients:
    def __init__(self, size):
        self.decay_x = np.zeros(size)
        self.decay_y = np.zeros(size)
        self.decay_z = np.zeros(size)
        self.diff_x = np.zeros(size)
        self.diff_y = np.zeros(size)
        self.diff_z = np.zeros(size)
        self.is_internal = np.zeros(size)
