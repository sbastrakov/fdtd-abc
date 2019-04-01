from grid.scalarfield import Scalar_field

import numpy as np

class YeeGrid:
    """Values of field components on a Yee grid. E values on middles of edges, B values on middles of faces"""

    def __init__(self, min_position, max_position, num_internal_cells, num_guard_cells_left, num_guard_cells_right):
        """Takes the box min and max position (internal area, without guard), size of internal area,
        size of guard on each side. Total number of cells is num_internal_cells + num_guard_cells_left + num_guard_cells_right
        """
        self.steps = (max_position - min_position) / num_internal_cells
        self.num_internal_cells = num_internal_cells
        self.num_guard_cells_left = num_guard_cells_left
        self.num_guard_cells_right = num_guard_cells_right
        self.num_cells = self.num_internal_cells + self.num_guard_cells_left + self.num_guard_cells_right
        ## self.min_position and max_position are with account for guard
        self.min_position = min_position - self.num_guard_cells_left * self.steps
        self.max_position = max_position + self.num_guard_cells_right * self.steps
        self.ex = self._create_component([0.5, 0.0, 0.0])
        self.ey = self._create_component([0.0, 0.5, 0.0])
        self.ez = self._create_component([0.0, 0.0, 0.5])
        self.bx = self._create_component([0.0, 0.5, 0.5])
        self.by = self._create_component([0.5, 0.0, 0.5])
        self.bz = self._create_component([0.5, 0.5, 0.0])

    def _create_component(self, shift):
        """Create a field component with a given shift inside a cell, shift is given as list, each component in 0..1 range"""
        return Scalar_field(self.min_position, self.max_position, self.num_cells, np.array(shift))
