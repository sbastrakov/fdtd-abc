import numpy as np
import scipy.constants
from scipy.constants import codata

from pml.data.coefficients import Coefficients
from pml.data.fields import Fields as SplitFields

import math


class Solver:
    """
    Yee's FDTD solver with split-field PML
    Operates on Yee grid
    All units in SI
    """

    def __init__(self,  num_pml_cells_left, num_pml_cells_right, order = 4, exponential_time_stepping = True):
        self.num_pml_cells_left = num_pml_cells_left
        self.num_pml_cells_right = num_pml_cells_right
        self.order = order
        self.exponential_time_stepping = exponential_time_stepping
        self._initialized = False

    def get_guard_size(self, num_internal_cells):
        guard_size_left = self.num_pml_cells_left
        guard_size_left[num_internal_cells == 1] = 0
        guard_size_right = self.num_pml_cells_right
        guard_size_right[num_internal_cells == 1] = 0
        return guard_size_left, guard_size_right

    def run_iteration(self, grid, dt):
        # Late initialization, currently only works if grid is the same all the time
        if not self._initialized:
            self._init(grid, dt)
        self.update_b(grid, 0.5 * dt)
        self.update_e(grid, dt)
        self.update_b(grid, 0.5 * dt)

    def _init(self, grid, dt):
        # for 1d and 2d cases disable PML along the fake dimensions
        self.num_pml_cells_left[grid.num_cells == 1] = 0
        self.num_pml_cells_right[grid.num_cells == 1] = 0
        self._init_max_sigma(grid)
        self._e_split = SplitFields(grid.ex, grid.ey, grid.ez)
        self._b_split = SplitFields(grid.bx, grid.by, grid.bz)
        self._init_coeffs(grid, dt)
        self._initialized = True

    def _init_max_sigma(self, grid):
        opt_sigma = self._get_opt_sigma(grid)
        sigma_opt_ratio = 1.0
        self.max_sigma = sigma_opt_ratio * opt_sigma

    # this value is for sigma E
    def _get_opt_sigma(self, grid):
        opt_sigma = np.array([0.0, 0.0, 0.0])
        ## equation (15) in CONVOLUTION PML (CPML): AN EFFICIENT FDTD IMPLEMENTATION OF THE CFS – PML FOR ARBITRARY MEDIA
        ## the same is (17) in  Performance advantages of CPML over UPML absorbing boundary conditions in FDTD algorithm
        for d in range(3):
            opt_sigma[d] = 0.8 * (self.order + 1) / (scipy.constants.value('characteristic impedance of vacuum') * grid.steps[d])
        return opt_sigma


    def _init_coeffs(self, grid, dt):
        self._e_coeffs = Coefficients(grid.num_cells)
        self._compute_coeffs(grid, np.array([0.0, 0.0, 0.0]), self._e_coeffs, dt)
        self._b_coeffs = Coefficients(grid.num_cells)
        self._compute_coeffs(grid, np.array([0.5, 0.5, 0.5]), self._b_coeffs, 0.5 * dt)

    def _compute_coeffs(self, grid, shift, coeffs, dt):
        for i in range(grid.num_cells[0]):
            for j in range(grid.num_cells[1]):
                for k in range(grid.num_cells[2]):
                    sigma_index = np.array([i, j, k]) + shift
                    sigma = self._get_sigma(grid, sigma_index)
                    sigma_normalized = sigma / scipy.constants.epsilon_0 # due to normalization can use for both E and B
                    # Coefficients for a simple scheme with no exponential stepping
                    decay_coeff = (1.0 - 0.5 * sigma_normalized * dt) / (1.0 + 0.5 * sigma_normalized * dt)
                    diff_coeff = np.array([dt, dt, dt]) / (1.0 + 0.5 * sigma_normalized * dt)
                    coeffs.is_internal[i, j, k] = self._is_internal(grid, sigma_index)
                    for d in range(3):
                        if sigma[d]:
                            if self.exponential_time_stepping:
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

    """This coefficient grows from 0 at PML-internal border to 1 at PML-external border"""
    def _get_depth_coeff(self, grid, index):
        coeff = np.array([0.0, 0.0, 0.0])
        for d in range(0, 3):
            coeff[d] = 0.0
            if index[d] < self.num_pml_cells_left[d]:
                coeff[d] = float(self.num_pml_cells_left[d] - index[d]) / self.num_pml_cells_left[d]
            if index[d] > grid.num_cells[d] - self.num_pml_cells_right[d]:
                coeff[d] = float(index[d] - grid.num_cells[d] + self.num_pml_cells_right[d]) / self.num_pml_cells_right[d]
            if coeff[d] < 0.0:
                coeff[d] = 0.0
        return coeff

    # returns sigma for E
    def _get_sigma(self, grid, index):
        """Index is float 3d array, values normalized to cell size"""
        depth_coeff = self._get_depth_coeff(grid, index)
        grading_coeff = np.power(depth_coeff, self.order)
        return self.max_sigma * grading_coeff

    def _is_internal(self, grid, index):
        depth_coeff = self._get_depth_coeff(grid, index)
        if np.array_equal(depth_coeff, np.array([0.0, 0.0, 0.0])):
            return 1.0
        else:
            return 0.0

    def update_e(self, grid, dt):
        for i in range(0, grid.num_cells[0]):
            for j in range(0, grid.num_cells[1]):
                for k in range(0, grid.num_cells[2]):
                    self.update_e_element(grid, dt, i, j, k)

    def update_e_element(self, grid, dt, i, j, k):
        # Discretized partial derivatives of magnetic field (same as in standard FDTD)
        dx = grid.steps[0]
        dy = grid.steps[1]
        dz = grid.steps[2]
        i_prev = (i - 1 + grid.num_cells[0]) % grid.num_cells[0]
        j_prev = (j - 1 + grid.num_cells[1]) % grid.num_cells[1]
        k_prev = (k - 1 + grid.num_cells[2]) % grid.num_cells[2]
        dbx_dy = (grid.bx[i, j, k] - grid.bx[i, j_prev, k]) / dy
        dbx_dz = (grid.bx[i, j, k] - grid.bx[i, j, k_prev]) / dz
        dby_dx = (grid.by[i, j, k] - grid.by[i_prev, j, k]) / dx
        dby_dz = (grid.by[i, j, k] - grid.by[i, j, k_prev]) / dz
        dbz_dx = (grid.bz[i, j, k] - grid.bz[i_prev, j, k]) / dx
        dbz_dy = (grid.bz[i, j, k] - grid.bz[i, j_prev, k]) / dy

        # special case for boundary indexes in PML: the external field values are zero
        if (i == 0) and (self.num_pml_cells_left[0] > 0):
            dby_dx = (grid.by[i, j, k] - 0.0) / dx
            dbz_dx = (grid.bz[i, j, k] - 0.0) / dx
        if (j == 0) and (self.num_pml_cells_left[1] > 0):
            dbx_dy = (grid.bx[i, j, k] - 0.0) / dy
            dbz_dy = (grid.bz[i, j, k] - 0.0) / dy
        if (k == 0) and (self.num_pml_cells_left[2] > 0):
            dbx_dz = (grid.bx[i, j, k] - 0.0) / dz
            dby_dz = (grid.by[i, j, k] - 0.0) / dz

        c2 = 1.0 / (scipy.constants.epsilon_0 * scipy.constants.mu_0)
        coeffs = self._e_coeffs
        if coeffs.is_internal[i, j, k]:
            # Standard Yee's scheme
            coeff = dt * c2
            grid.ex[i, j, k] += coeff * (dbz_dy - dby_dz)
            grid.ey[i, j, k] += coeff * (dbx_dz - dbz_dx)
            grid.ez[i, j, k] += coeff * (dby_dx - dbx_dy)
        else:
            # Update split fields
            e = self._e_split
            e.yx[i, j, k] = coeffs.decay_x[i, j, k] * e.yx[i, j, k] - coeffs.diff_x[i, j, k] * c2 * dbz_dx
            e.zx[i, j, k] = coeffs.decay_x[i, j, k] * e.zx[i, j, k] + coeffs.diff_x[i, j, k] * c2 * dby_dx
            e.xy[i, j, k] = coeffs.decay_y[i, j, k] * e.xy[i, j, k] + coeffs.diff_y[i, j, k] * c2 * dbz_dy
            e.zy[i, j, k] = coeffs.decay_y[i, j, k] * e.zy[i, j, k] - coeffs.diff_y[i, j, k] * c2 * dbx_dy
            e.xz[i, j, k] = coeffs.decay_z[i, j, k] * e.xz[i, j, k] - coeffs.diff_z[i, j, k] * c2 * dby_dz
            e.yz[i, j, k] = coeffs.decay_z[i, j, k] * e.yz[i, j, k] + coeffs.diff_z[i, j, k] * c2 * dbx_dz
            # Sum up split fields to get full fields
            grid.ex[i, j, k] = e.xy[i, j, k] + e.xz[i, j, k]
            grid.ey[i, j, k] = e.yx[i, j, k] + e.yz[i, j, k]
            grid.ez[i, j, k] = e.zx[i, j, k] + e.zy[i, j, k]

    def update_b(self, grid, dt):
        for i in range(0, grid.num_cells[0]):
            for j in range(0, grid.num_cells[1]):
                for k in range(0, grid.num_cells[2]):
                    self.update_b_element(grid, dt, i, j, k)

    def update_b_element(self, grid, dt, i, j, k):
        # Discretized partial derivatives of electric field (same as in standard FDTD)
        # with periodic boundaries
        dx = grid.steps[0]
        dy = grid.steps[1]
        dz = grid.steps[2]
        i_next = (i + 1) % grid.num_cells[0]
        j_next = (j + 1) % grid.num_cells[1]
        k_next = (k + 1) % grid.num_cells[2]
        dex_dy = (grid.ex[i, j_next, k] - grid.ex[i, j, k]) / dy
        dex_dz = (grid.ex[i, j, k_next] - grid.ex[i, j, k]) / dz
        dey_dx = (grid.ey[i_next, j, k] - grid.ey[i, j, k]) / dx
        dey_dz = (grid.ey[i, j, k_next] - grid.ey[i, j, k]) / dz
        dez_dx = (grid.ez[i_next, j, k] - grid.ez[i, j, k]) / dx
        dez_dy = (grid.ez[i, j_next, k] - grid.ez[i, j, k]) / dy

        # special case for boundary indexes in PML: the external field values are zero
        if (i == grid.num_cells[0] - 1) and (self.num_pml_cells_right[0] > 0):
            dey_dx = (0.0 - grid.ey[i, j, k]) / dx
            dez_dx = (0.0 - grid.ez[i, j, k]) / dx
        if (j == grid.num_cells[1] - 1) and (self.num_pml_cells_right[1] > 0):
            dex_dy = (0.0 - grid.ex[i, j, k]) / dy
            dez_dy = (0.0 - grid.ez[i, j, k]) / dy
        if (k == grid.num_cells[2] - 1) and (self.num_pml_cells_right[2] > 0):
            dex_dz = (0.0 - grid.ex[i, j, k]) / dz
            dey_dz = (0.0 - grid.ey[i, j, k]) / dz

        coeffs = self._b_coeffs
        if coeffs.is_internal[i, j, k]:
            # Standard Yee's scheme
            coeff = dt
            grid.bx[i, j, k] += coeff * (dey_dz - dez_dy)
            grid.by[i, j, k] += coeff * (dez_dx - dex_dz)
            grid.bz[i, j, k] += coeff * (dex_dy - dey_dx)
        else:
            # Update split fields
            b = self._b_split
            b.yx[i, j, k] = coeffs.decay_x[i, j, k] * b.yx[i, j, k] + coeffs.diff_x[i, j, k] * dez_dx
            b.zx[i, j, k] = coeffs.decay_x[i, j, k] * b.zx[i, j, k] - coeffs.diff_x[i, j, k] * dey_dx
            b.xy[i, j, k] = coeffs.decay_y[i, j, k] * b.xy[i, j, k] - coeffs.diff_y[i, j, k] * dez_dy
            b.zy[i, j, k] = coeffs.decay_y[i, j, k] * b.zy[i, j, k] + coeffs.diff_y[i, j, k] * dex_dy
            b.xz[i, j, k] = coeffs.decay_z[i, j, k] * b.xz[i, j, k] + coeffs.diff_z[i, j, k] * dey_dz
            b.yz[i, j, k] = coeffs.decay_z[i, j, k] * b.yz[i, j, k] - coeffs.diff_z[i, j, k] * dex_dz
            # Sum up split fields to get full fields
            grid.bx[i, j, k] = b.xy[i, j, k] + b.xz[i, j, k]
            grid.by[i, j, k] = b.yx[i, j, k] + b.yz[i, j, k]
            grid.bz[i, j, k] = b.zx[i, j, k] + b.zy[i, j, k]

