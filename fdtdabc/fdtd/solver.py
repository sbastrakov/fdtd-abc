import numpy as np
import scipy.constants


class Solver:
    """
    Yee's FDTD solver with periodic boundary conditions
    Operates on Yee grid
    All units in SI
    """

    def get_guard_size(self, num_internal_cells):
        return np.array([0, 0, 0]), np.array([0, 0, 0])

    def run_iteration(self, grid, dt):
        # At start and end of the iterations E and B are given at the same time
        # (same as used in PIC), so iteration is split in halfB - fullE - halfB
        self.e_coeff_x = dt / (scipy.constants.epsilon_0 * scipy.constants.mu_0) / grid.steps[0]
        self.e_coeff_y = dt / (scipy.constants.epsilon_0 * scipy.constants.mu_0) / grid.steps[1]
        self.b_coeff_x = -dt / grid.steps[0]
        self.b_coeff_y = -dt / grid.steps[1]
        self.update_b(grid, 0.5 * dt)
        self.update_e(grid, dt)
        self.update_b(grid, 0.5 * dt)

    def update_e(self, grid, dt):
        i = 1
        while i < grid.num_cells[0]:
            j = 1
            while j < grid.num_cells[1]:
                self.update_e_element(grid, dt, i, j, 0)
                j += 1
            i += 1
        #i_range = range(grid.num_cells[0])
        #j_range = range(grid.num_cells[1])
        #for i in i_range:
        #    for j in j_range:
        #        self.update_e_element(grid, dt, i, j, 0)
                #for k in range(grid.num_cells[2]):
                #    self.update_e_element(grid, dt, i, j, k)

    def update_e_element(self, grid, dt, i, j, k):
        # Yee's scheme: E_new = E_old + dt/(eps0 * mu0) * rot(B) = E_old + coeff * rot(B)
        grid.ex[i, j, k] += self.e_coeff_y * (grid.bz[i, j, k] - grid.bz[i, j - 1, k])
        grid.ey[i, j, k] -= self.e_coeff_x * (grid.bz[i, j, k] - grid.bz[i - 1, j, k])
        grid.ez[i, j, k] += self.e_coeff_x * (grid.by[i, j, k] - grid.by[i - 1, j, k]) - self.e_coeff_y * (grid.bx[i, j, k] - grid.bx[i, j - 1, k])

    def update_b(self, grid, dt):
        i = 0
        while i < grid.num_cells[0] - 1:
            j = 0
            while j < grid.num_cells[1] - 1:
                self.update_b_element(grid, dt, i, j, 0)
                j += 1
            i += 1
                #for k in range(grid.num_cells[2]):
                #    self.update_b_element(grid, dt, i, j, k)

    def update_b_element(self, grid, dt, i, j, k):
        # Yee's scheme: B_new = B_old - dt * rot(E) = B_old + coeff * rot(E)
        grid.bx[i, j, k] += self.b_coeff_y * (grid.ez[i, j + 1, k] - grid.ez[i, j, k])
        grid.by[i, j, k] -= self.b_coeff_x * (grid.ez[i + 1, j, k] - grid.ez[i, j, k])
        grid.bz[i, j, k] += self.b_coeff_x * (grid.ey[i + 1, j, k] - grid.ey[i, j, k]) - self.b_coeff_y * (grid.ex[i, j + 1, k] - grid.ex[i, j, k])
