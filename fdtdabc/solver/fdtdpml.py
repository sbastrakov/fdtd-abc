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
            self._init(grid)
        self.update_b(grid, 0.5 * dt)
        self.update_e(grid, dt)
        self.update_b(grid, 0.5 * dt)

    def _init(self, grid):
        self.num_cells = grid.num_cells
        self.pml_width = self.num_pml_cells * grid.steps
        # Compute max absorption
        r0 = 1e-8 # basic relative reflection
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
        self._initialized = True

    def _create_split_field(self, full_field):
        split_field = copy.deepcopy(full_field)
        split_field.values.fill(0.0)
        return split_field

    def update_e(self, grid, dt):
        for i in range(1, grid.num_cells[0]):
            for j in range(1, grid.num_cells[1]):
                for k in range(1, grid.num_cells[2]):
                    self.update_e_element(grid, dt, i, j, k)

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

    def update_e_element(self, grid, dt, i, j, k):
        c = 29979245800.0 # cm / s
        cdt = c * dt
        dx = grid.steps[0]
        dy = grid.steps[1]
        dz = grid.steps[2]

        # decay and derivative coefficients for each direction (not field component)
        sigma_index = np.array([i, j, k])
        sigma = self._get_sigma(sigma_index)
        decay_coeff = np.exp(-sigma * cdt)
        diff_coeff = np.array([cdt, cdt, cdt])
        for d in range(3):
            if sigma[d]:
                diff_coeff[d] = (1.0 - decay_coeff[d]) / sigma[d]

        # Discretized partial derivatives of magnetic field (same as in standard FDTD)
        dbx_dy = (grid.bx[i, j, k] - grid.bx[i, j - 1, k]) / dy
        dbx_dz = (grid.bx[i, j, k] - grid.bx[i, j, k - 1]) / dz
        dby_dx = (grid.by[i, j, k] - grid.by[i - 1, j, k]) / dx
        dby_dz = (grid.by[i, j, k] - grid.by[i, j, k - 1]) / dz
        dbz_dx = (grid.bz[i, j, k] - grid.bz[i - 1, j, k]) / dx
        dbz_dy = (grid.bz[i, j, k] - grid.bz[i, j - 1, k]) / dy

        is_internal_area = ((sigma[0] == 0.0) and (sigma[1] == 0.0) and (sigma[2] == 0.0))
        ##print("(" + str(i) + ", " + str(j) + ", " + str(k) + "): sigmaE = " + str(sigma) + ", internal = " + str(is_internal_area))
        if is_internal_area:
            # Standard Yee's scheme
            grid.ex[i, j, k] += cdt * (dbz_dy - dby_dz)
            grid.ey[i, j, k] += cdt * (dbx_dz - dbz_dx)
            grid.ez[i, j, k] += cdt * (dby_dx - dbx_dy)
        else:
            # Update split fields
            self.eyx[i, j, k] = decay_coeff[0] * self.eyx[i, j, k] - diff_coeff[0] * dbz_dx
            self.ezx[i, j, k] = decay_coeff[0] * self.ezx[i, j, k] + diff_coeff[0] * dby_dx
            self.exy[i, j, k] = decay_coeff[1] * self.exy[i, j, k] + diff_coeff[1] * dbz_dy
            self.ezy[i, j, k] = decay_coeff[1] * self.ezy[i, j, k] - diff_coeff[1] * dbx_dy
            self.exz[i, j, k] = decay_coeff[2] * self.exz[i, j, k] - diff_coeff[2] * dby_dz
            self.eyz[i, j, k] = decay_coeff[2] * self.eyz[i, j, k] + diff_coeff[2] * dbx_dz
            # Sum up split fields to get full fields
            grid.ex[i, j, k] = self.exy[i, j, k] + self.exz[i, j, k]
            grid.ey[i, j, k] = self.eyx[i, j, k] + self.eyz[i, j, k]
            grid.ez[i, j, k] = self.ezx[i, j, k] + self.ezy[i, j, k]

    def update_b(self, grid, dt):
        for i in range(grid.num_cells[0] - 1):
            for j in range(grid.num_cells[1] - 1):
                for k in range(grid.num_cells[2] - 1):
                    self.update_b_element(grid, dt, i, j, k)

    def update_b_element(self, grid, dt, i, j, k):
        c = 29979245800.0 # cm / s
        cdt = c * dt
        dx = grid.steps[0]
        dy = grid.steps[1]
        dz = grid.steps[2]

        # decay and derivative coefficients for each direction (not field component)
        sigma_index = np.array([i + 0.5, j + 0.5, k + 0.5])
        sigma = self._get_sigma(sigma_index)
        decay_coeff = np.exp(-sigma * cdt)
        diff_coeff = np.array([cdt, cdt, cdt])
        for d in range(3):
            if sigma[d]:
                diff_coeff[d] = (1.0 - decay_coeff[d]) / sigma[d]

        # Discretized partial derivatives of electric field (same as in standard FDTD)
        dex_dy = (grid.ex[i, j + 1, k] - grid.ex[i, j, k]) / dy
        dex_dz = (grid.ex[i, j, k + 1] - grid.ex[i, j, k]) / dz
        dey_dx = (grid.ey[i + 1, j, k] - grid.ey[i, j, k]) / dx
        dey_dz = (grid.ey[i, j, k + 1] - grid.ey[i, j, k]) / dz
        dez_dx = (grid.ez[i + 1, j, k] - grid.ez[i, j, k]) / dx
        dez_dy = (grid.ez[i, j + 1, k] - grid.ez[i, j, k]) / dy

        is_internal_area = ((sigma[0] == 0.0) and (sigma[1] == 0.0) and (sigma[2] == 0.0))
        ##print("(" + str(i) + ", " + str(j) + ", " + str(k) + "): sigmaE = " + str(sigma) + ", internal = " + str(is_internal_area))
        if is_internal_area:
            # Standard Yee's scheme
            grid.bx[i, j, k] += cdt * (dey_dz - dez_dy)
            grid.by[i, j, k] += cdt * (dez_dx - dex_dz)
            grid.bz[i, j, k] += cdt * (dex_dy - dey_dx)
        else:
            # Update split fields
            self.byx[i, j, k] = decay_coeff[0] * self.byx[i, j, k] + diff_coeff[0] * dez_dx
            self.bzx[i, j, k] = decay_coeff[0] * self.bzx[i, j, k] - diff_coeff[0] * dey_dx
            self.bxy[i, j, k] = decay_coeff[1] * self.bxy[i, j, k] - diff_coeff[1] * dez_dy
            self.bzy[i, j, k] = decay_coeff[1] * self.bzy[i, j, k] + diff_coeff[1] * dex_dy
            self.bxz[i, j, k] = decay_coeff[2] * self.bxz[i, j, k] + diff_coeff[2] * dey_dz
            self.byz[i, j, k] = decay_coeff[2] * self.byz[i, j, k] - diff_coeff[2] * dex_dz
            # Sum up split fields to get full fields
            grid.bx[i, j, k] = self.bxy[i, j, k] + self.bxz[i, j, k]
            grid.by[i, j, k] = self.byx[i, j, k] + self.byz[i, j, k]
            grid.bz[i, j, k] = self.bzx[i, j, k] + self.bzy[i, j, k]

