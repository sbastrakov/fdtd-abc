import numpy as np
import scipy.constants


class Solver:
    """
    Yee's FDTD solver with periodic boundary conditions
    Operates on Yee grid
    All units in SI
    """

    def get_guard_size(self, num_internal_cells):
        return np.array([0, 0, 0])

    def run_iteration(self, grid, dt):
        # At start and end of the iterations E and B are given at the same time
        # (same as used in PIC), so iteration is split in halfB - fullE - halfB
        self.update_b(grid, 0.5 * dt)
        self.update_e(grid, dt)
        self.update_b(grid, 0.5 * dt)

    def update_e(self, grid, dt):
        for i in range(grid.num_cells[0]):
            for j in range(grid.num_cells[1]):
                for k in range(grid.num_cells[2]):
                    self.update_e_element(grid, dt, i, j, k)

    def update_e_element(self, grid, dt, i, j, k):
        # Discretized partial derivatives of magnetic field (indexing is done to match grid.Yee_grid)
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

        # Yee's scheme: E_new = E_old + dt/(eps0 * mu0) * rot(B) = E_old + coeff * rot(B)
        coeff = dt / (scipy.constants.epsilon_0 * scipy.constants.mu_0) # also, coeff = dt * c^2, as c^2 = 1/(eps0 * mu0)
        grid.ex[i, j, k] += coeff * (dbz_dy - dby_dz)
        grid.ey[i, j, k] += coeff * (dbx_dz - dbz_dx)
        grid.ez[i, j, k] += coeff * (dby_dx - dbx_dy)

    def update_b(self, grid, dt):
        for i in range(grid.num_cells[0]):
            for j in range(grid.num_cells[1]):
                for k in range(grid.num_cells[2]):
                    self.update_b_element(grid, dt, i, j, k)

    def update_b_element(self, grid, dt, i, j, k):
        # Discretized partial derivatives of electric field (indexing is done to match grid.Yee_grid)
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

        # Yee's scheme: B_new = B_old - dt * rot(E) = B_old + coeff * rot(E)
        coeff = -dt
        grid.bx[i, j, k] += coeff * (dez_dy - dey_dz)
        grid.by[i, j, k] += coeff * (dex_dz - dez_dx)
        grid.bz[i, j, k] += coeff * (dey_dx - dex_dy)
