import numpy as np
import math

class FdtdPML:
    """Yee FDTD solver with PML, operates on Yee grid with split fields stored, uses CGS units"""

    def __init__(self, num_pml_cells, grid_steps, order = 4):
        self.num_pml_cells = num_pml_cells
        self.order = order
        self.grid_steps = grid_steps
        r0 = 1e-8 # basic relative reflection
        self.max_sigma = np.array([0.0, 0.0, 0.0])
        self.pml_width = np.array([self.num_pml_cells[0] * self.grid_steps[0], self.num_pml_cells[1] * self.grid_steps[1], self.num_pml_cells[2] * self.grid_steps[2]])
        for d in range(3):
            if self.num_pml_cells[d]:
                self.max_sigma[d] = -math.log(r0) * (self.order + 1) / (2 * self.pml_width[d])


    def run_iteration(self, grid):
        # At start and end of the iterations E and B are given at the same time
        # (same as used in PIC), so iteration is split in halfB - fullE - halfB
        self.update_b(grid, 0.5 * grid.dt)
        self.update_e(grid, grid.dt)
        self.update_b(grid, 0.5 * grid.dt)

    def update_e(self, grid, dt):
        for i in range(1, grid.num_cells[0]):
            for j in range(1, grid.num_cells[1]):
                for k in range(1, grid.num_cells[2]):
                    self.update_e_element(grid, dt, i, j, k)

    def _get_sigma(i, j, k):
        # This needs to be a polynomial growth from 0 at border with internal area to max_sigma at outer border
        return self.max_sigma

    def update_e_element(self, grid, dt, i, j, k):
        c = 29979245800.0 # cm / s
        cdt = c * dt
        dx = grid.steps[0]
        dy = grid.steps[1]
        dz = grid.steps[2]

        sigma = self._get_sigma(i, j, k)
        decay_coeff = np.array([math.exp(-sigma[0] * cdt), math.exp(-sigma[1] * cdt), math.exp(-sigma[2] * cdt)])
        diff_coeff = np.array([(decay_coeff[0] - 1.0) / (sigma[0] * dx), (decay_coeff[1] - 1.0) / (sigma[1] * dy), (decay_coeff[2] - 1.0) / (sigma[2] * dz)])

        grid.eyx[i, j, k] = decay_coeff[0] * grid.eyx[i, j, k] + diff_coeff[0] * (grid.bz[i, j, k] - grid.bz[i - 1, j, k])
        grid.ezx[i, j, k] = decay_coeff[0] * grid.ezx[i, j, k] - diff_coeff[0] * (grid.by[i, j, k] - grid.by[i - 1, j, k])
        grid.exy[i, j, k] = decay_coeff[1] * grid.exy[i, j, k] - diff_coeff[1] * (grid.bz[i, j, k] - grid.bz[i, j - 1, k])
        grid.ezy[i, j, k] = decay_coeff[1] * grid.ezy[i, j, k] + diff_coeff[1] * (grid.bx[i, j, k] - grid.bx[i, j - 1, k])
        grid.exz[i, j, k] = decay_coeff[2] * grid.exz[i, j, k] + diff_coeff[2] * (grid.by[i, j, k] - grid.by[i, j, k - 1])
        grid.eyz[i, j, k] = decay_coeff[2] * grid.exz[i, j, k] + diff_coeff[2] * (grid.bx[i, j, k] - grid.bx[i, j, k - 1])

        grid.ex[i, j, k] = grid.exy[i, j, k] + grid.exz[i, j, k]
        grid.ey[i, j, k] = grid.eyx[i, j, k] + grid.eyz[i, j, k]
        grid.ez[i, j, k] = grid.ezx[i, j, k] + grid.ezy[i, j, k]

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

        # Update similar to E

        grid.bx[i, j, k] = grid.bxy[i, j, k] + grid.bxz[i, j, k]
        grid.by[i, j, k] = grid.byx[i, j, k] + grid.byz[i, j, k]
        grid.bz[i, j, k] = grid.bzx[i, j, k] + grid.bzy[i, j, k]
