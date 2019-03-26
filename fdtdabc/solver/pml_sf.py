import numpy as np
import scipy.constants
from scipy.constants import codata

import copy
import math


class PML_SF:
    """
    Yee's FDTD solver with split-field PML
    Operates on Yee grid
    All units in SI
    """

    def __init__(self,  num_pml_cells_left, num_pml_cells_right, order = 4, exponential_time_stepping = True):
        self.num_pml_cells_left = num_pml_cells_left
        self.num_pml_cells_right = num_pml_cells_right
        self.order = 4
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
        # Initialize split fields
        self.exy = self._create_split_field(grid.ex)
        self.exz = self._create_split_field(grid.ex)
        self.eyx = self._create_split_field(grid.ey)
        self.eyz = self._create_split_field(grid.ey)
        self.ezx = self._create_split_field(grid.ez)
        self.ezy = self._create_split_field(grid.ez)
        self.bxy = self._create_split_field(grid.bx)
        self.bxz = self._create_split_field(grid.bx)
        self.byx = self._create_split_field(grid.by)
        self.byz = self._create_split_field(grid.by)
        self.bzx = self._create_split_field(grid.bz)
        self.bzy = self._create_split_field(grid.bz)
        self._init_coeffs(grid, dt)
        self._initialized = True

    def _init_max_sigma(self, grid):
        opt_sigma = self._get_opt_sigma(grid)
        sigma_opt_ratio = 1.0
        self.max_sigma = sigma_opt_ratio * opt_sigma

    # this value is for sigma/eps0 = sigma*/mu0
    def _get_opt_sigma(self, grid):
        opt_sigma = np.array([0.0, 0.0, 0.0])
        for d in range(3):
            opt_sigma[d] = 0.8 * (self.order + 1) * scipy.constants.c / grid.steps[d]
            ## equation (15) divided by eps0 in CONVOLUTION PML (CPML): AN EFFICIENT FDTD IMPLEMENTATION OF THE CFS – PML FOR ARBITRARY MEDIA
            ## basically the same is (17) in  Performance advantages of CPML over UPML absorbing boundary conditions in FDTD algorithm
        return opt_sigma

    def _create_split_field(self, full_field):
        split_field = copy.deepcopy(full_field)
        split_field.values.fill(0.0)
        return split_field

    def _init_coeffs(self, grid, dt):
        self._e_decay_coeff_x = np.zeros(grid.num_cells)
        self._e_decay_coeff_y = np.zeros(grid.num_cells)
        self._e_decay_coeff_z = np.zeros(grid.num_cells)
        self._e_diff_coeff_x = np.zeros(grid.num_cells)
        self._e_diff_coeff_y = np.zeros(grid.num_cells)
        self._e_diff_coeff_z = np.zeros(grid.num_cells)
        self._e_is_internal = np.zeros(grid.num_cells)
        self._b_decay_coeff_x = np.zeros(grid.num_cells)
        self._b_decay_coeff_y = np.zeros(grid.num_cells)
        self._b_decay_coeff_z = np.zeros(grid.num_cells)
        self._b_diff_coeff_x = np.zeros(grid.num_cells)
        self._b_diff_coeff_y = np.zeros(grid.num_cells)
        self._b_diff_coeff_z = np.zeros(grid.num_cells)
        self._b_is_internal = np.zeros(grid.num_cells)
        self._init_coeffs_field(grid, np.array([0.0, 0.0, 0.0]), self._e_decay_coeff_x, self._e_decay_coeff_y, self._e_decay_coeff_z, self._e_diff_coeff_x, self._e_diff_coeff_y, self._e_diff_coeff_z, self._e_is_internal, dt)
        self._init_coeffs_field(grid, np.array([0.5, 0.5, 0.5]), self._b_decay_coeff_x, self._b_decay_coeff_y, self._b_decay_coeff_z, self._b_diff_coeff_x, self._b_diff_coeff_y, self._b_diff_coeff_z, self._b_is_internal, 0.5 * dt)

    def _init_coeffs_field(self, grid, shift, decay_coeff_x, decay_coeff_y, decay_coeff_z, diff_coeff_x, diff_coeff_y, diff_coeff_z, is_internal, dt):
        for i in range(grid.num_cells[0]):
            for j in range(grid.num_cells[1]):
                for k in range(grid.num_cells[2]):
                    sigma_index = np.array([i, j, k]) + shift
                    sigma = self._get_sigma(grid, sigma_index)
                    decay_coeff = (1.0 - 0.5 * sigma * dt) / (1.0 + 0.5 * sigma * dt)
                    diff_coeff = np.array([dt, dt, dt]) / (1.0 + 0.5 * sigma * dt)
                    is_internal[i, j, k] = 1.0
                    for d in range(3):
                        if sigma[d]:
                            is_internal[i, j, k] = 0.0
                            if self.exponential_time_stepping:
                                decay_coeff[d] = math.exp(-sigma[d] * dt)
                                diff_coeff[d] = (1.0 - decay_coeff[d]) / sigma[d]
                    decay_coeff_x[i, j, k] = decay_coeff[0]
                    decay_coeff_y[i, j, k] = decay_coeff[1]
                    decay_coeff_z[i, j, k] = decay_coeff[2]
                    diff_coeff_x[i, j, k] = diff_coeff[0]
                    diff_coeff_y[i, j, k] = diff_coeff[1]
                    diff_coeff_z[i, j, k] = diff_coeff[2]

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

    # returns sigma/eps0 = sigma*/mu0
    def _get_sigma(self, grid, index):
        """Index is float 3d array, values normalized to cell size"""
        depth_coeff = self._get_depth_coeff(grid, index)
        grading_coeff = np.power(depth_coeff, self.order)
        return self.max_sigma * grading_coeff

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

        if self._e_is_internal[i, j, k]:
            # Standard Yee's scheme
            coeff = dt / (scipy.constants.epsilon_0 * scipy.constants.mu_0) # also, coeff = dt * c^2, as c^2 = 1/(eps0 * mu0)
            grid.ex[i, j, k] += coeff * (dbz_dy - dby_dz)
            grid.ey[i, j, k] += coeff * (dbx_dz - dbz_dx)
            grid.ez[i, j, k] += coeff * (dby_dx - dbx_dy)
        else:
            # Update split fields
            #print("E " + str([i, j, k]) + ": decay = "  + str(self._e_decay_coeff_x[i, j, k]) + ", k = " + str(coeff_k[0]) + ", diff = " + str(self._e_diff_coeff_x[i, j, k]))
            self.eyx[i, j, k] = self._e_decay_coeff_x[i, j, k] * self.eyx[i, j, k] - self._e_diff_coeff_x[i, j, k] * dbz_dx / (scipy.constants.epsilon_0 * scipy.constants.mu_0)
            self.ezx[i, j, k] = self._e_decay_coeff_x[i, j, k] * self.ezx[i, j, k] + self._e_diff_coeff_x[i, j, k] * dby_dx / (scipy.constants.epsilon_0 * scipy.constants.mu_0)
            self.exy[i, j, k] = self._e_decay_coeff_y[i, j, k] * self.exy[i, j, k] + self._e_diff_coeff_y[i, j, k] * dbz_dy / (scipy.constants.epsilon_0 * scipy.constants.mu_0)
            self.ezy[i, j, k] = self._e_decay_coeff_y[i, j, k] * self.ezy[i, j, k] - self._e_diff_coeff_y[i, j, k] * dbx_dy / (scipy.constants.epsilon_0 * scipy.constants.mu_0)
            self.exz[i, j, k] = self._e_decay_coeff_z[i, j, k] * self.exz[i, j, k] - self._e_diff_coeff_z[i, j, k] * dby_dz / (scipy.constants.epsilon_0 * scipy.constants.mu_0)
            self.eyz[i, j, k] = self._e_decay_coeff_z[i, j, k] * self.eyz[i, j, k] + self._e_diff_coeff_z[i, j, k] * dbx_dz / (scipy.constants.epsilon_0 * scipy.constants.mu_0)
            # Sum up split fields to get full fields
            grid.ex[i, j, k] = self.exy[i, j, k] + self.exz[i, j, k]
            grid.ey[i, j, k] = self.eyx[i, j, k] + self.eyz[i, j, k]
            grid.ez[i, j, k] = self.ezx[i, j, k] + self.ezy[i, j, k]

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

        if self._b_is_internal[i, j, k]:
            # Standard Yee's scheme
            coeff = -dt
            grid.bx[i, j, k] += coeff * (dez_dy - dey_dz)
            grid.by[i, j, k] += coeff * (dex_dz - dez_dx)
            grid.bz[i, j, k] += coeff * (dey_dx - dex_dy)
        else:
            # Update split fields
            self.byx[i, j, k] = self._b_decay_coeff_x[i, j, k] * self.byx[i, j, k] + self._b_diff_coeff_x[i, j, k] * dez_dx
            self.bzx[i, j, k] = self._b_decay_coeff_x[i, j, k] * self.bzx[i, j, k] - self._b_diff_coeff_x[i, j, k] * dey_dx
            self.bxy[i, j, k] = self._b_decay_coeff_y[i, j, k] * self.bxy[i, j, k] - self._b_diff_coeff_y[i, j, k] * dez_dy
            self.bzy[i, j, k] = self._b_decay_coeff_y[i, j, k] * self.bzy[i, j, k] + self._b_diff_coeff_y[i, j, k] * dex_dy
            self.bxz[i, j, k] = self._b_decay_coeff_z[i, j, k] * self.bxz[i, j, k] + self._b_diff_coeff_z[i, j, k] * dey_dz
            self.byz[i, j, k] = self._b_decay_coeff_z[i, j, k] * self.byz[i, j, k] - self._b_diff_coeff_z[i, j, k] * dex_dz
            # Sum up split fields to get full fields
            grid.bx[i, j, k] = self.bxy[i, j, k] + self.bxz[i, j, k]
            grid.by[i, j, k] = self.byx[i, j, k] + self.byz[i, j, k]
            grid.bz[i, j, k] = self.bzx[i, j, k] + self.bzy[i, j, k]

