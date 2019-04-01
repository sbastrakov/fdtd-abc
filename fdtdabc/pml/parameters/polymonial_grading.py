import numpy as np


class Parameter:
    """Polynomially graded parameter of PML"""

    def __init__(self, grid, num_pml_cells_left, num_pml_cells_right, inner_value, outer_value , order, grow_inner_to_outer):
        self._grid = grid
        self._num_pml_cells_left = num_pml_cells_left
        self._num_pml_cells_right = num_pml_cells_right
        self._inner_value = inner_value
        self._outer_value = outer_value
        self._order = order
        self._grow_inner_to_outer = grow_inner_to_outer

    def get(self, index):
        """Index is float 3d array, values normalized to cell size"""
        depth_coeff = self._get_depth_coeff(index)
        min_value = self._inner_value
        max_value = self._outer_value
        if not self._grow_inner_to_outer:
            depth_coeff = 1.0 - depth_coeff
            min_value = self._outer_value
            max_value = self._inner_value
        grading_coeff = np.power(depth_coeff, self._order)
        return min_value + (max_value - min_value) * grading_coeff

    """This coefficient grows from 0 at PML-internal border to 1 at PML-external border"""
    def _get_depth_coeff(self, index):
        coeff = np.array([0.0, 0.0, 0.0])
        for d in range(0, 3):
            coeff[d] = 0.0
            if index[d] < self._num_pml_cells_left[d]:
                coeff[d] = float(self._num_pml_cells_left[d] - index[d]) / self._num_pml_cells_left[d]
            if index[d] > self._grid.num_cells[d] - self._num_pml_cells_right[d]:
                coeff[d] = float(index[d] - self._grid.num_cells[d] + self._num_pml_cells_right[d]) / self._num_pml_cells_right[d]
            if coeff[d] < 0.0:
                coeff[d] = 0.0
        return coeff