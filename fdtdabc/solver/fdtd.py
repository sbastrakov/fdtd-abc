
class Fdtd:
    """Yee FDTD solver, operates on Yee grid, uses CGS units"""
    
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

    def update_e_element(self, grid, dt, i, j, k):
        c = 29979245800.0 # cm / s
        cdt = c * dt
        dx = grid.steps[0]
        dy = grid.steps[1]
        dz = grid.steps[2]
        grid.ex[i, j, k] += cdt * ((grid.bz[i, j, k] - grid.bz[i, j - 1, k]) / dy - (grid.by[i, j, k] - grid.by[i, j, k - 1]) / dz)
        grid.ey[i, j, k] += cdt * ((grid.bx[i, j, k] - grid.bx[i, j, k - 1]) / dz - (grid.bz[i, j, k] - grid.bz[i - 1, j, k]) / dx)
        grid.ez[i, j, k] += cdt * ((grid.by[i, j, k] - grid.by[i - 1, j, k]) / dx - (grid.bx[i, j, k] - grid.bx[i, j - 1, k]) / dy)

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
        grid.bx[i, j, k] += cdt * ((grid.ey[i, j, k + 1] - grid.ey[i, j, k]) / dz - (grid.ez[i, j + 1, k] - grid.ez[i, j, k]) / dy)
        grid.by[i, j, k] += cdt * ((grid.ez[i + 1, j, k] - grid.ez[i, j, k]) / dx - (grid.ex[i, j, k + 1] - grid.ex[i, j, k]) / dz)
        grid.bz[i, j, k] += cdt * ((grid.ex[i, j + 1, k] - grid.ex[i, j, k]) / dy - (grid.ey[i + 1, j, k] - grid.ey[i, j, k]) / dx)
