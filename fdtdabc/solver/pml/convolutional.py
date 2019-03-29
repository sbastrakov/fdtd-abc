import numpy as np
import scipy.constants
from scipy.constants import codata

import copy
import math


class Solver:
    """Yee FDTD solver with split-field PML, operates on Yee grid with split fields stored, uses CGS units
    Implementation and notation are based on the following paper:
    Branko D. Gvozdic, Dusan Z. Djurdjevic. Performance advantages of CPML over UPML absorbing boundary conditions in FDTD algorithm.
    Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53
     https://www.degruyter.com/downloadpdf/j/jee.2017.68.issue-1/jee-2017-0006/jee-2017-0006.pdf
    """

    def __init__(self, num_pml_cells_left, num_pml_cells_right, order = 4):
        self.num_pml_cells_left = num_pml_cells_left
        self.num_pml_cells_right = num_pml_cells_right
        self.order = order # for sigma and kappa
        self.order_alpha = 1.0 # like in the paper
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
        self._init_max_params(grid)
        # Initialize split fields
        self.psi_exy = self._create_split_field(grid.ex)
        self.psi_exz = self._create_split_field(grid.ex)
        self.psi_eyx = self._create_split_field(grid.ey)
        self.psi_eyz = self._create_split_field(grid.ey)
        self.psi_ezx = self._create_split_field(grid.ez)
        self.psi_ezy = self._create_split_field(grid.ez)
        self.psi_bxy = self._create_split_field(grid.bx)
        self.psi_bxz = self._create_split_field(grid.bx)
        self.psi_byx = self._create_split_field(grid.by)
        self.psi_byz = self._create_split_field(grid.by)
        self.psi_bzx = self._create_split_field(grid.bz)
        self.psi_bzy = self._create_split_field(grid.bz)
        self._init_coeffs(grid, dt)
        self._initialized = True

    def _init_max_params(self, grid):
        opt_sigma = self._get_opt_sigma(grid)
        sigma_opt_ratio = 1.0
        self.max_sigma = sigma_opt_ratio * opt_sigma
        max_kappa_value = 15.0 # 15.0 # like in the paper
        self.max_kappa = np.array([max_kappa_value, max_kappa_value, max_kappa_value])
        max_alpha_value = 0.24 # like in the paper
        self.max_alpha = np.array([max_alpha_value, max_alpha_value, max_alpha_value])

    # this value is for sigma E, not normalized
    def _get_opt_sigma(self, grid):
        z0 = scipy.constants.value('characteristic impedance of vacuum')
        opt_sigma = np.array([0.0, 0.0, 0.0])
        for d in range(3):
            opt_sigma[d] = 0.8 * (self.order + 1) / (z0 * grid.steps[d])
            ## equation (15) in CONVOLUTION PML (CPML): AN EFFICIENT FDTD IMPLEMENTATION OF THE CFS – PML FOR ARBITRARY MEDIA
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
        self._init_coeffs_field(grid , np.array([0.0, 0.0, 0.0]), self._e_decay_coeff_x, self._e_decay_coeff_y, self._e_decay_coeff_z, self._e_diff_coeff_x, self._e_diff_coeff_y, self._e_diff_coeff_z, self._e_is_internal, dt)
        self._init_coeffs_field(grid, np.array([0.5, 0.5, 0.5]), self._b_decay_coeff_x, self._b_decay_coeff_y, self._b_decay_coeff_z, self._b_diff_coeff_x, self._b_diff_coeff_y, self._b_diff_coeff_z, self._b_is_internal, 0.5 * dt)

    def _init_coeffs_field(self, grid, shift, decay_coeff_x, decay_coeff_y, decay_coeff_z, diff_coeff_x, diff_coeff_y, diff_coeff_z, is_internal, dt):
        eps0 = scipy.constants.epsilon_0
        for i in range(grid.num_cells[0]):
            for j in range(grid.num_cells[1]):
                for k in range(grid.num_cells[2]):
                    index = np.array([i, j, k]) + shift
                    sigma = self._get_sigma(grid, index)
                    kappa = self._get_kappa(grid, index)
                    alpha = self._get_alpha(grid, index)
                    psi_b = [1.0, 1.0, 1.0]
                    psi_c = [0.0, 0.0, 0.0]
                    is_internal[i, j, k] = self._is_internal(grid, index)
                    for d in range(3):
                        psi_b[d] = math.exp(-(sigma[d] / kappa[d] + alpha[d]) * dt / eps0)
                        if sigma[d] + alpha[d] * kappa[d] != 0.0:
                            psi_c[d] = sigma[d] * (psi_b[d] - 1.0) / (kappa[d] * (sigma[d] + alpha[d] * kappa[d]))
                    decay_coeff_x[i, j, k] = psi_b[0]
                    decay_coeff_y[i, j, k] = psi_b[1]
                    decay_coeff_z[i, j, k] = psi_b[2]
                    diff_coeff_x[i, j, k] = psi_c[0]
                    diff_coeff_y[i, j, k] = psi_c[1]
                    diff_coeff_z[i, j, k] = psi_c[2]

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

    def _get_kappa(self, grid, index):
        depth_coeff = self._get_depth_coeff(grid, index)
        grading_coeff = np.power(depth_coeff, self.order)
        return 1.0 + (self.max_kappa - 1.0) * grading_coeff

    def _get_alpha(self, grid, index):
        depth_coeff = 1.0 - self._get_depth_coeff(grid, index) # alpha decreases with depth in PML
        grading_coeff = np.power(depth_coeff, self.order_alpha)
        return self.max_alpha * grading_coeff

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

        if self._e_is_internal[i, j, k]:
            # Standard Yee's scheme
            coeff = dt / (scipy.constants.epsilon_0 * scipy.constants.mu_0) # also, coeff = dt * c^2, as c^2 = 1/(eps0 * mu0)
            grid.ex[i, j, k] += coeff * (dbz_dy - dby_dz)
            grid.ey[i, j, k] += coeff * (dbx_dz - dbz_dx)
            grid.ez[i, j, k] += coeff * (dby_dx - dbx_dy)
        else:
            # Update psi components
            index = np.array([i, j, k])
            kappa = self._get_kappa(grid, index)
            #print("E " + str([i, j, k]) + ": decay = "  + str(self._e_decay_coeff_x[i, j, k]) + ", k = " + str(coeff_k[0]) + ", diff = " + str(self._e_diff_coeff_x[i, j, k]))
            self.psi_eyx[i, j, k] = self._e_decay_coeff_x[i, j, k] * self.psi_eyx[i, j, k] + self._e_diff_coeff_x[i, j, k] * dbz_dx
            self.psi_ezx[i, j, k] = self._e_decay_coeff_x[i, j, k] * self.psi_ezx[i, j, k] + self._e_diff_coeff_x[i, j, k] * dby_dx
            self.psi_exy[i, j, k] = self._e_decay_coeff_y[i, j, k] * self.psi_exy[i, j, k] + self._e_diff_coeff_y[i, j, k] * dbz_dy
            self.psi_ezy[i, j, k] = self._e_decay_coeff_y[i, j, k] * self.psi_ezy[i, j, k] + self._e_diff_coeff_y[i, j, k] * dbx_dy
            self.psi_exz[i, j, k] = self._e_decay_coeff_z[i, j, k] * self.psi_exz[i, j, k] + self._e_diff_coeff_z[i, j, k] * dby_dz
            self.psi_eyz[i, j, k] = self._e_decay_coeff_z[i, j, k] * self.psi_eyz[i, j, k] + self._e_diff_coeff_z[i, j, k] * dbx_dz
            # Update fields
            coeff = dt / (scipy.constants.epsilon_0 * scipy.constants.mu_0)
            grid.ex[i, j, k] += coeff * (dbz_dy / kappa[1] - dby_dz / kappa[2] + self.psi_exy[i, j, k] - self.psi_exz[i, j, k])
            grid.ey[i, j, k] += coeff * (dbx_dz / kappa[2] - dbz_dx / kappa[0] + self.psi_eyz[i, j, k] - self.psi_eyx[i, j, k])
            grid.ez[i, j, k] += coeff * (dby_dx / kappa[0] - dbx_dy / kappa[1] + self.psi_ezx[i, j, k] - self.psi_ezy[i, j, k])

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
            # Update psi components
            index = np.array([i + 0.5, j + 0.5, k + 0.5])
            kappa = self._get_kappa(grid, index) # called k in the paper, renamed to not confuse with index
            self.psi_byx[i, j, k] = self._b_decay_coeff_x[i, j, k] * self.psi_byx[i, j, k] + self._b_diff_coeff_x[i, j, k] * dez_dx
            self.psi_bzx[i, j, k] = self._b_decay_coeff_x[i, j, k] * self.psi_bzx[i, j, k] + self._b_diff_coeff_x[i, j, k] * dey_dx
            self.psi_bxy[i, j, k] = self._b_decay_coeff_y[i, j, k] * self.psi_bxy[i, j, k] + self._b_diff_coeff_y[i, j, k] * dez_dy
            self.psi_bzy[i, j, k] = self._b_decay_coeff_y[i, j, k] * self.psi_bzy[i, j, k] + self._b_diff_coeff_y[i, j, k] * dex_dy
            self.psi_bxz[i, j, k] = self._b_decay_coeff_z[i, j, k] * self.psi_bxz[i, j, k] + self._b_diff_coeff_z[i, j, k] * dey_dz
            self.psi_byz[i, j, k] = self._b_decay_coeff_z[i, j, k] * self.psi_byz[i, j, k] + self._b_diff_coeff_z[i, j, k] * dex_dz
            # Update fields
            grid.bx[i, j, k] += dt * (dey_dz / kappa[2] - dez_dy / kappa[1] + self.psi_bxz[i, j, k] - self.psi_bxy[i, j, k])
            grid.by[i, j, k] += dt * (dez_dx / kappa[0] - dex_dz / kappa[2] + self.psi_byx[i, j, k] - self.psi_byz[i, j, k])
            grid.bz[i, j, k] += dt * (dex_dy / kappa[1] - dey_dx / kappa[0] + self.psi_bzy[i, j, k] - self.psi_bzx[i, j, k])

