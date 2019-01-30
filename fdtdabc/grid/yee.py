from grid.scalarfield import Scalar_field

import numpy as np

class Yee_grid:
    """Values of field components on a Yee grid"""

    def __init__(self, dt, min_position, max_position, num_cells):
        self.dt = dt
        self.min_position = min_position
        self.max_position = max_position
        self.num_cells = num_cells
        self.steps = (max_position - min_position) / num_cells
        self.ex = Scalar_field(min_position, max_position, num_cells, np.array([0.5, 0.0, 0.0]))
        self.ey = Scalar_field(min_position, max_position, num_cells, np.array([0.0, 0.5, 0.0]))
        self.ez = Scalar_field(min_position, max_position, num_cells, np.array([0.0, 0.0, 0.5]))
        self.bx = Scalar_field(min_position, max_position, num_cells, np.array([0.0, 0.5, 0.5]))
        self.by = Scalar_field(min_position, max_position, num_cells, np.array([0.5, 0.0, 0.5]))
        self.bz = Scalar_field(min_position, max_position, num_cells, np.array([0.5, 0.5, 0.0]))

        ## these are only for PML (could store only in PML area, but for simplicity store everywhere)
        self.exy = Scalar_field(min_position, max_position, num_cells, np.array([0.5, 0.0, 0.0]))
        self.exz = Scalar_field(min_position, max_position, num_cells, np.array([0.5, 0.0, 0.0]))
        self.eyx = Scalar_field(min_position, max_position, num_cells, np.array([0.0, 0.5, 0.0]))
        self.eyz = Scalar_field(min_position, max_position, num_cells, np.array([0.0, 0.5, 0.0]))
        self.ezx = Scalar_field(min_position, max_position, num_cells, np.array([0.0, 0.0, 0.5]))
        self.ezy = Scalar_field(min_position, max_position, num_cells, np.array([0.0, 0.0, 0.5]))
        self.bxy = Scalar_field(min_position, max_position, num_cells, np.array([0.0, 0.5, 0.5]))
        self.bxz = Scalar_field(min_position, max_position, num_cells, np.array([0.0, 0.5, 0.5]))
        self.byx = Scalar_field(min_position, max_position, num_cells, np.array([0.5, 0.0, 0.5]))
        self.byz = Scalar_field(min_position, max_position, num_cells, np.array([0.5, 0.0, 0.5]))
        self.bzx = Scalar_field(min_position, max_position, num_cells, np.array([0.5, 0.5, 0.0]))
        self.bzy = Scalar_field(min_position, max_position, num_cells, np.array([0.5, 0.5, 0.0]))                