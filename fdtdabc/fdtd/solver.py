import numpy as np
import scipy.constants


class Solver:
    """
    Yee's FDTD solver with periodic boundary conditions
    Operates on Yee grid
    All units in SI
    """

    def __init__(self, periodic = True):
        self.periodic = periodic

    def get_guard_size(self, num_internal_cells):
        return np.array([0, 0, 0]), np.array([0, 0, 0])

    def run_iteration(self, grid, dt):
        # At start and end of the iterations E and B are given at the same time
        # (same as used in PIC), so iteration is split in halfB - fullE - halfB
        self.e_coeff_x = dt / (scipy.constants.epsilon_0 * scipy.constants.mu_0) / grid.steps[0]
        self.e_coeff_y = dt / (scipy.constants.epsilon_0 * scipy.constants.mu_0) / grid.steps[1]
        self.e_coeff_z = dt / (scipy.constants.epsilon_0 * scipy.constants.mu_0) / grid.steps[2]
        self.b_coeff_x = 0.5 * dt / grid.steps[0]
        self.b_coeff_y = 0.5 * dt / grid.steps[1]
        self.b_coeff_z = 0.5 * dt / grid.steps[2]
        self.update_b(grid, 0.5 * dt)
        self.update_e(grid, dt)
        self.update_b(grid, 0.5 * dt)

    def update_e(self, grid, dt):
        i_range = self._get_e_range(grid.num_cells[0])
        j_range = self._get_e_range(grid.num_cells[1])
        k_range = self._get_e_range(grid.num_cells[2])
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    self.update_e_element(grid, dt, i, j, k)

    def _get_e_range(self, num_cells):
        if (not self.periodic) and (num_cells > 1):
            return range(1, num_cells)
        else:
            return range(num_cells)

    def update_e_element(self, grid, dt, i, j, k):
        # Yee's scheme: E_new = E_old + dt/(eps0 * mu0) * rot(B)
        # negative indices are interpreted as periodic boundaries
        grid.ex[i, j, k] += self.e_coeff_y * (grid.bz[i, j, k] - grid.bz[i, j - 1, k]) - self.e_coeff_z * (grid.by[i, j, k] - grid.by[i, j, k - 1])
        grid.ey[i, j, k] += self.e_coeff_z * (grid.bx[i, j, k] - grid.bx[i, j, k - 1]) - self.e_coeff_x * (grid.bz[i, j, k] - grid.bz[i - 1, j, k])
        grid.ez[i, j, k] += self.e_coeff_x * (grid.by[i, j, k] - grid.by[i - 1, j, k]) - self.e_coeff_y * (grid.bx[i, j, k] - grid.bx[i, j - 1, k])


    def update_b(self, grid, dt):
        i_range = self._get_b_range(grid.num_cells[0])
        j_range = self._get_b_range(grid.num_cells[1])
        k_range = self._get_b_range(grid.num_cells[2])
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    self.update_b_element(grid, dt, i, j, k)

    def _get_b_range(self, num_cells):
        if (not self.periodic) and (num_cells > 1):
            return range(num_cells - 1)
        else:
            return range(num_cells)

    def update_b_element(self, grid, dt, i, j, k):
        # Yee's scheme: B_new = B_old - dt * rot(E)
        # indices over the array size are interpreted as periodic boundaries
        i_next = (i + 1) % grid.num_cells[0]
        j_next = (j + 1) % grid.num_cells[1]
        k_next = (k + 1) % grid.num_cells[2]
        grid.bx[i, j, k] += self.b_coeff_z * (grid.ey[i, j, k_next] - grid.ey[i, j, k]) - self.b_coeff_y * (grid.ez[i, j_next, k] - grid.ez[i, j, k])
        grid.by[i, j, k] += self.b_coeff_x * (grid.ez[i_next, j, k] - grid.ez[i, j, k]) - self.b_coeff_z * (grid.ex[i, j, k_next] - grid.ex[i, j, k])
        grid.bz[i, j, k] += self.b_coeff_y * (grid.ex[i, j_next, k] - grid.ex[i, j, k]) - self.b_coeff_x * (grid.ey[i_next, j, k] - grid.ey[i, j, k])
