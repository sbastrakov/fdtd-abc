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
        self.pml_width = self.num_pml_cells * grid.steps
        # Compute max absorption
        r0 = 1e-8 # basic relative reflection
        self.max_sigma = np.array([0.0, 0.0, 0.0])
        for d in range(3):
            if self.num_pml_cells[d]:
                self.max_sigma[d] = -math.log(r0) * (self.order + 1) / (2 * self.pml_width[d])
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

    def _get_sigma(self, i, j, k):
        # This needs to be a polynomial growth from 0 at border with internal area to max_sigma at outer border
        sigma = np.array([0.0, 0.0, 0.0])
        return self.max_sigma

    def update_e_element(self, grid, dt, i, j, k):
        c = 29979245800.0 # cm / s
        cdt = c * dt
        dx = grid.steps[0]
        dy = grid.steps[1]
        dz = grid.steps[2]

        sigma = self._get_sigma(i, j, k)

        # decay and derivative coefficients for each direction (not field component)
        decay_coeff = np.exp(-sigma * cdt)
        diff_coeff = np.array([cdt, cdt, cdt])
        for d in range(3):
            if sigma[d]:
                diff_coeff[d] = (decay_coeff[d] - 1.0) / sigma[d]

        # Discretized partial derivatives of magnetic field (same as in standard FDTD)
        dbx_dy = (grid.bx[i, j, k] - grid.bx[i, j - 1, k]) / dy
        dbx_dz = (grid.bx[i, j, k] - grid.bx[i, j, k - 1]) / dz
        dby_dx = (grid.by[i, j, k] - grid.by[i - 1, j, k]) / dx
        dby_dz = (grid.by[i, j, k] - grid.by[i, j, k - 1]) / dz
        dbz_dx = (grid.bz[i, j, k] - grid.bz[i - 1, j, k]) / dx
        dbz_dy = (grid.bz[i, j, k] - grid.bz[i, j - 1, k]) / dy

        # Update split fields
        self.eyx[i, j, k] = decay_coeff[0] * self.eyx[i, j, k] + diff_coeff[0] * dbz_dx
        self.ezx[i, j, k] = decay_coeff[0] * self.ezx[i, j, k] - diff_coeff[0] * dby_dx
        self.exy[i, j, k] = decay_coeff[1] * self.exy[i, j, k] - diff_coeff[1] * dbz_dy
        self.ezy[i, j, k] = decay_coeff[1] * self.ezy[i, j, k] + diff_coeff[1] * dbx_dy
        self.exz[i, j, k] = decay_coeff[2] * self.exz[i, j, k] + diff_coeff[2] * dby_dz
        self.eyz[i, j, k] = decay_coeff[2] * self.exz[i, j, k] + diff_coeff[2] * dbx_dz

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
