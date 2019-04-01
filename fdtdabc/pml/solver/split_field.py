import numpy as np
import scipy.constants

from pml.data.coefficients import Coefficients
from pml.data.fields import Fields as SplitFields
from pml.parameters.polymonial_grading import Parameter
from pml.parameters.sigma_max import get_sigma_max
from pml.solver.base import Solver as SolverBase

import math


class Solver(SolverBase):
    """
    Yee's FDTD solver with split-field PML
    Operates on Yee grid
    All units in SI
    """

    def __init__(self,  num_pml_cells_left, num_pml_cells_right, order = 4, exponential_time_stepping = True):
        SolverBase.__init__(self, num_pml_cells_left, num_pml_cells_right)
        self.order = order
        self.exponential_time_stepping = exponential_time_stepping

    def _init_coeffs(self, grid, dt):
        sigma_min = np.zeros(3)
        sigma_max = get_sigma_max(grid.steps, self.order)
        self._sigma = Parameter(grid, self.num_pml_cells_left, self.num_pml_cells_right, sigma_min, sigma_max, self.order, True)
        self._is_internal = Parameter(grid, self.num_pml_cells_left, self.num_pml_cells_right, np.zeros(3), np.ones(3), 1, True)
        SolverBase._init_coeffs(self, grid, dt)

    def _compute_coeff(self, coeffs, i, j, k, shift, dt):
        index = np.array([i, j, k]) + shift
        sigma = self._sigma.get(index)
        sigma_normalized = sigma / scipy.constants.epsilon_0 # due to normalization can use for both E and B
        # Coefficients for a simple scheme with no exponential stepping
        decay_coeff = (1.0 - 0.5 * sigma_normalized * dt) / (1.0 + 0.5 * sigma_normalized * dt)
        diff_coeff = np.array([dt, dt, dt]) / (1.0 + 0.5 * sigma_normalized * dt)
        coeffs.is_internal[i, j, k] = np.array_equal(self._is_internal.get(index), np.zeros(3))
        for d in range(3):
            if self.exponential_time_stepping and sigma[d]:
                # Coefficients for exponential stepping based on (3.49) in Taflove 1st ed.
                # diff_coeff has reversed sign to become positive
                # and uses sigma_normalized and not sigma in the denominator
                decay_coeff[d] = math.exp(-sigma_normalized[d] * dt)
                diff_coeff[d] = (1.0 - decay_coeff[d]) / sigma_normalized[d]
        coeffs.decay_x[i, j, k] = decay_coeff[0]
        coeffs.decay_y[i, j, k] = decay_coeff[1]
        coeffs.decay_z[i, j, k] = decay_coeff[2]
        coeffs.diff_x[i, j, k] = diff_coeff[0]
        coeffs.diff_y[i, j, k] = diff_coeff[1]
        coeffs.diff_z[i, j, k] = diff_coeff[2]

    def _update_pml_e_element(self, grid, dt, i, j, k, dbx_dy, dbx_dz, dby_dx, dby_dz, dbz_dx, dbz_dy):
        c2 = 1.0 / (scipy.constants.epsilon_0 * scipy.constants.mu_0)
        coeffs = self._e_coeffs
        e = self._e_split
        e.yx[i, j, k] = coeffs.decay_x[i, j, k] * e.yx[i, j, k] - coeffs.diff_x[i, j, k] * c2 * dbz_dx
        e.zx[i, j, k] = coeffs.decay_x[i, j, k] * e.zx[i, j, k] + coeffs.diff_x[i, j, k] * c2 * dby_dx
        e.xy[i, j, k] = coeffs.decay_y[i, j, k] * e.xy[i, j, k] + coeffs.diff_y[i, j, k] * c2 * dbz_dy
        e.zy[i, j, k] = coeffs.decay_y[i, j, k] * e.zy[i, j, k] - coeffs.diff_y[i, j, k] * c2 * dbx_dy
        e.xz[i, j, k] = coeffs.decay_z[i, j, k] * e.xz[i, j, k] - coeffs.diff_z[i, j, k] * c2 * dby_dz
        e.yz[i, j, k] = coeffs.decay_z[i, j, k] * e.yz[i, j, k] + coeffs.diff_z[i, j, k] * c2 * dbx_dz
        grid.ex[i, j, k] = e.xy[i, j, k] + e.xz[i, j, k]
        grid.ey[i, j, k] = e.yx[i, j, k] + e.yz[i, j, k]
        grid.ez[i, j, k] = e.zx[i, j, k] + e.zy[i, j, k]

    def _update_pml_b_element(self, grid, dt, i, j, k, dex_dy, dex_dz, dey_dx, dey_dz, dez_dx, dez_dy):
        coeffs = self._b_coeffs
        b = self._b_split
        b.yx[i, j, k] = coeffs.decay_x[i, j, k] * b.yx[i, j, k] + coeffs.diff_x[i, j, k] * dez_dx
        b.zx[i, j, k] = coeffs.decay_x[i, j, k] * b.zx[i, j, k] - coeffs.diff_x[i, j, k] * dey_dx
        b.xy[i, j, k] = coeffs.decay_y[i, j, k] * b.xy[i, j, k] - coeffs.diff_y[i, j, k] * dez_dy
        b.zy[i, j, k] = coeffs.decay_y[i, j, k] * b.zy[i, j, k] + coeffs.diff_y[i, j, k] * dex_dy
        b.xz[i, j, k] = coeffs.decay_z[i, j, k] * b.xz[i, j, k] + coeffs.diff_z[i, j, k] * dey_dz
        b.yz[i, j, k] = coeffs.decay_z[i, j, k] * b.yz[i, j, k] - coeffs.diff_z[i, j, k] * dex_dz
        grid.bx[i, j, k] = b.xy[i, j, k] + b.xz[i, j, k]
        grid.by[i, j, k] = b.yx[i, j, k] + b.yz[i, j, k]
        grid.bz[i, j, k] = b.zx[i, j, k] + b.zy[i, j, k]
