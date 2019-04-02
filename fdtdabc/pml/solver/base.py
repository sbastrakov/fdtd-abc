import numpy as np
import scipy.constants

from pml.data.coefficients import Coefficients
from pml.data.fields import Fields as SplitFields

import math


class Solver:
    """
    Base class for PML solvers
    Implements general solver workflow and logic of separating PML and internal area
    Updates values for the internal area and calls hooks for PML
    All units are in SI
    """

    def __init__(self,  num_pml_cells_left, num_pml_cells_right):
        self.num_pml_cells_left = num_pml_cells_left
        self.num_pml_cells_right = num_pml_cells_right
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
        self._e_split = SplitFields(grid.ex, grid.ey, grid.ez)
        self._b_split = SplitFields(grid.bx, grid.by, grid.bz)
        self._init_coeffs(grid, dt)
        self._initialized = True

    def _init_coeffs(self, grid, dt):
        self._e_coeffs = Coefficients(grid.num_cells)
        self._compute_coeffs(grid, np.array([0.0, 0.0, 0.0]), self._e_coeffs, dt)
        self._b_coeffs = Coefficients(grid.num_cells)
        self._compute_coeffs(grid, np.array([0.5, 0.5, 0.5]), self._b_coeffs, 0.5 * dt)

    def _compute_coeffs(self, grid, shift, coeffs, dt):
        for i in range(grid.num_cells[0]):
            for j in range(grid.num_cells[1]):
                for k in range(grid.num_cells[2]):
                    self._compute_coeff(coeffs, i, j, k, shift, dt)

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
        # note to self: python does this with -1 index automatically
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
        # note to self: maybe only set a split component to 0 and not the full component
        if (i == 0) and (self.num_pml_cells_left[0] > 0):
            dby_dx = (grid.by[i, j, k] - 0.0) / dx
            dbz_dx = (grid.bz[i, j, k] - 0.0) / dx
        if (j == 0) and (self.num_pml_cells_left[1] > 0):
            dbx_dy = (grid.bx[i, j, k] - 0.0) / dy
            dbz_dy = (grid.bz[i, j, k] - 0.0) / dy
        if (k == 0) and (self.num_pml_cells_left[2] > 0):
            dbx_dz = (grid.bx[i, j, k] - 0.0) / dz
            dby_dz = (grid.by[i, j, k] - 0.0) / dz

        coeffs = self._e_coeffs
        if coeffs.is_internal[i, j, k]:
            # Standard Yee's scheme
            c2 = 1.0 / (scipy.constants.epsilon_0 * scipy.constants.mu_0)
            coeff = dt * c2
            grid.ex[i, j, k] += coeff * (dbz_dy - dby_dz)
            grid.ey[i, j, k] += coeff * (dbx_dz - dbz_dx)
            grid.ez[i, j, k] += coeff * (dby_dx - dbx_dy)
        else:
            self._update_pml_e_element(grid, dt, i, j, k, dbx_dy, dbx_dz, dby_dx, dby_dz, dbz_dx, dbz_dy)

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
            self._update_pml_b_element(grid, dt, i, j, k, dex_dy, dex_dz, dey_dx, dey_dz, dez_dx, dez_dy)
