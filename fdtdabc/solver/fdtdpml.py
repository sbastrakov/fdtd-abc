import numpy as np

import copy
import math


class FdtdPML:
    """Yee FDTD solver with PML, operates on Yee grid with split fields stored, uses CGS units"""

    def __init__(self, num_pml_cells, order = 4):
        self.num_pml_cells = num_pml_cells
        self.order = order
        self._initialized = False

    def get_guard_size(self):
        return self.num_pml_cells

    def run_iteration(self, grid, dt):
        # Late initialization, currently only works if grid is the same all the time
        if not self._initialized:
            self._init(grid, dt)
        self.update_b(grid, 0.5 * dt)
        self.update_e(grid, dt)
        self.update_b(grid, 0.5 * dt)

    def _init(self, grid, dt):
        self.num_cells = grid.num_cells
        self.pml_width = self.num_pml_cells * grid.steps
        # Compute max absorption
        r0 = 1e-7 # basic relative reflection R(0)
        self.max_sigma = np.array([0.0, 0.0, 0.0])
        for d in range(3):
            if self.num_pml_cells[d]:
                self.max_sigma[d] = -math.log(r0) * (self.order + 1) / (2 * self.pml_width[d])
        ##print(self.max_sigma)
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
        c = 29979245800.0 # cm / s
        cdt = c * dt
        for i in range(grid.num_cells[0]):
            for j in range(grid.num_cells[1]):
                for k in range(grid.num_cells[2]):
                    sigma_index = np.array([i, j, k]) + shift
                    sigma = self._get_sigma(sigma_index)
                    decay_coeff = np.exp(-sigma * cdt)
                    diff_coeff = np.array([cdt, cdt, cdt])
                    is_internal[i, j, k] = 1.0
                    for d in range(3):
                        if sigma[d]:
                            diff_coeff[d] = (1.0 - decay_coeff[d]) / sigma[d]
                            is_internal[i, j, k] = 0.0
                    decay_coeff_x[i, j, k] = decay_coeff[0]
                    decay_coeff_y[i, j, k] = decay_coeff[1]
                    decay_coeff_z[i, j, k] = decay_coeff[2]
                    diff_coeff_x[i, j, k] = diff_coeff[0]
                    diff_coeff_y[i, j, k] = diff_coeff[1]
                    diff_coeff_z[i, j, k] = diff_coeff[2]

    def _get_sigma(self, index):
        """Index is float 3d array, values normalized to cell size"""
        # This needs to be a polynomial growth from 0 at border with internal area to max_sigma at outer border
        sigma = np.array([0.0, 0.0, 0.0])
        for d in range(0, 3):
            coeff = 0.0
            if index[d] < self.num_pml_cells[d]:
                coeff = float(self.num_pml_cells[d] - index[d]) / self.num_pml_cells[d]
            if index[d] > self.num_cells[d] - self.num_pml_cells[d]:
                coeff = float(index[d] - self.num_cells[d] + self.num_pml_cells[d]) / self.num_pml_cells[d]
            if coeff < 0.0:
                coeff = 0.0
            sigma[d] = self.max_sigma[d] * math.pow(coeff, self.order)
        return sigma

    def update_e(self, grid, dt):
        start_idx = np.array([0, 0, 0])
        start_idx[self.num_pml_cells > 0] = 1
        end_idx = grid.num_cells
        index_ranges = []
        for d in range(3):
            r = range(start_idx[d], end_idx[d])
            if grid.num_cells[d] == 1:
                r = range(1)
            index_ranges.append(r)
        for i in index_ranges[0]:
            for j in index_ranges[1]:
                for k in index_ranges[2]:
                    self.update_e_element(grid, dt, i, j, k)

    def update_e_element(self, grid, dt, i, j, k):
        c = 29979245800.0 # cm / s
        cdt = c * dt
        dx = grid.steps[0]
        dy = grid.steps[1]
        dz = grid.steps[2]

        # Discretized partial derivatives of magnetic field (same as in standard FDTD)
        i_prev = (i - 1 + grid.num_cells[0]) % grid.num_cells[0]
        j_prev = (j - 1 + grid.num_cells[1]) % grid.num_cells[1]
        k_prev = (k - 1 + grid.num_cells[2]) % grid.num_cells[2]
        dbx_dy = (grid.bx[i, j, k] - grid.bx[i, j_prev, k]) / dy
        dbx_dz = (grid.bx[i, j, k] - grid.bx[i, j, k_prev]) / dz
        dby_dx = (grid.by[i, j, k] - grid.by[i_prev, j, k]) / dx
        dby_dz = (grid.by[i, j, k] - grid.by[i, j, k_prev]) / dz
        dbz_dx = (grid.bz[i, j, k] - grid.bz[i_prev, j, k]) / dx
        dbz_dy = (grid.bz[i, j, k] - grid.bz[i, j_prev, k]) / dy

        if self._e_is_internal[i, j, k]:
            # Standard Yee's scheme
            grid.ex[i, j, k] += cdt * (dbz_dy - dby_dz)
            grid.ey[i, j, k] += cdt * (dbx_dz - dbz_dx)
            grid.ez[i, j, k] += cdt * (dby_dx - dbx_dy)
        else:
            # Update split fields
            self.eyx[i, j, k] = self._e_decay_coeff_x[i, j, k] * self.eyx[i, j, k] - self._e_diff_coeff_x[i, j, k] * dbz_dx
            self.ezx[i, j, k] = self._e_decay_coeff_x[i, j, k] * self.ezx[i, j, k] + self._e_diff_coeff_x[i, j, k] * dby_dx
            self.exy[i, j, k] = self._e_decay_coeff_y[i, j, k] * self.exy[i, j, k] + self._e_diff_coeff_y[i, j, k] * dbz_dy
            self.ezy[i, j, k] = self._e_decay_coeff_y[i, j, k] * self.ezy[i, j, k] - self._e_diff_coeff_y[i, j, k] * dbx_dy
            self.exz[i, j, k] = self._e_decay_coeff_z[i, j, k] * self.exz[i, j, k] - self._e_diff_coeff_z[i, j, k] * dby_dz
            self.eyz[i, j, k] = self._e_decay_coeff_z[i, j, k] * self.eyz[i, j, k] + self._e_diff_coeff_z[i, j, k] * dbx_dz
            # Sum up split fields to get full fields
            grid.ex[i, j, k] = self.exy[i, j, k] + self.exz[i, j, k]
            grid.ey[i, j, k] = self.eyx[i, j, k] + self.eyz[i, j, k]
            grid.ez[i, j, k] = self.ezx[i, j, k] + self.ezy[i, j, k]

    def update_b(self, grid, dt):
        start_idx = [0, 0, 0]
        end_idx = np.array(grid.num_cells)
        end_idx[self.num_pml_cells > 0] = end_idx[self.num_pml_cells > 0] - 1
        index_ranges = []
        for d in range(3):
            r = range(start_idx[d], end_idx[d])
            if grid.num_cells[d] == 1:
                r = range(1)
            index_ranges.append(r)
        for i in index_ranges[0]:
            for j in index_ranges[1]:
                for k in index_ranges[2]:
                    self.update_b_element(grid, dt, i, j, k)

    def update_b_element(self, grid, dt, i, j, k):
        c = 29979245800.0 # cm / s
        cdt = c * dt
        dx = grid.steps[0]
        dy = grid.steps[1]
        dz = grid.steps[2]

        # Discretized partial derivatives of electric field (same as in standard FDTD)
        i_next = (i + 1) % grid.num_cells[0]
        j_next = (j + 1) % grid.num_cells[1]
        k_next = (k + 1) % grid.num_cells[2]
        dex_dy = (grid.ex[i, j_next, k] - grid.ex[i, j, k]) / dy
        dex_dz = (grid.ex[i, j, k_next] - grid.ex[i, j, k]) / dz
        dey_dx = (grid.ey[i_next, j, k] - grid.ey[i, j, k]) / dx
        dey_dz = (grid.ey[i, j, k_next] - grid.ey[i, j, k]) / dz
        dez_dx = (grid.ez[i_next, j, k] - grid.ez[i, j, k]) / dx
        dez_dy = (grid.ez[i, j_next, k] - grid.ez[i, j, k]) / dy

        if self._b_is_internal[i, j, k]:
            # Standard Yee's scheme
            grid.bx[i, j, k] += cdt * (dey_dz - dez_dy)
            grid.by[i, j, k] += cdt * (dez_dx - dex_dz)
            grid.bz[i, j, k] += cdt * (dex_dy - dey_dx)
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

