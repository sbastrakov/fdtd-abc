import numpy as np

class Scalar_field:
    """Grid values of a single field component"""

    def __init__(self, min_position, max_position, num_cells, shift):
        self.min_position = min_position
        self.max_position = max_position
        self.num_cells = num_cells
        self.steps = (max_position - min_position) / num_cells
        self.shift = shift
        self.values = np.zeros(num_cells)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def position(self, index):
        return self.min_position + (self.shift + index) * self.steps
