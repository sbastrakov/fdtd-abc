
class Fdtd:
    """Yee FDTD solver, operates on Yee grid, uses CGS units"""

    def get_guard_size(self):
        return [1, 1, 1]

    def run_iteration(self, grid, dt):
        # At start and end of the iterations E and B are given at the same time
        # (same as used in PIC), so iteration is split in halfB - fullE - halfB
        self.update_b(grid, 0.5 * dt)
        self.update_e(grid, dt)
        self.update_b(grid, 0.5 * dt)

    def update_e(self, grid, dt):
        guard_size = self.get_guard_size()
        start_idx = guard_size
        end_idx = grid.num_cells - guard_size
        for i in range(start_idx[0], end_idx[0]):
            for j in range(start_idx[1], end_idx[1]):
                for k in range(start_idx[2], end_idx[2]):
                    self.update_e_element(grid, dt, i, j, k)
        self.apply_e_bc(grid)

    def update_e_element(self, grid, dt, i, j, k):
        c = 29979245800.0 # cm / s
        cdt = c * dt
        dx = grid.steps[0]
        dy = grid.steps[1]
        dz = grid.steps[2]

        # Discretized partial derivatives of magnetic field (indexing is done to match grid.Yee_grid)
        dbx_dy = (grid.bx[i, j, k] - grid.bx[i, j - 1, k]) / dy
        dbx_dz = (grid.bx[i, j, k] - grid.bx[i, j, k - 1]) / dz
        dby_dx = (grid.by[i, j, k] - grid.by[i - 1, j, k]) / dx
        dby_dz = (grid.by[i, j, k] - grid.by[i, j, k - 1]) / dz
        dbz_dx = (grid.bz[i, j, k] - grid.bz[i - 1, j, k]) / dx
        dbz_dy = (grid.bz[i, j, k] - grid.bz[i, j - 1, k]) / dy

        # Yee's scheme
        grid.ex[i, j, k] += cdt * (dbz_dy - dby_dz)
        grid.ey[i, j, k] += cdt * (dbx_dz - dbz_dx)
        grid.ez[i, j, k] += cdt * (dby_dx - dbx_dy)

    def apply_e_bc(self, grid):
        """Apply periodic boundary conditions for the electric field"""
        self._apply_component_bc(grid.ex, grid.num_internal_cells, grid.num_guard_cells)
        self._apply_component_bc(grid.ey, grid.num_internal_cells, grid.num_guard_cells)
        self._apply_component_bc(grid.ez, grid.num_internal_cells, grid.num_guard_cells)

    def update_b(self, grid, dt):
        guard_size = self.get_guard_size()
        start_idx = guard_size
        end_idx = grid.num_cells - guard_size
        for i in range(start_idx[0], end_idx[0]):
            for j in range(start_idx[1], end_idx[1]):
                for k in range(start_idx[2], end_idx[2]):
                    self.update_b_element(grid, dt, i, j, k)
        self.apply_b_bc(grid)

    def update_b_element(self, grid, dt, i, j, k):
        c = 29979245800.0 # cm / s
        cdt = c * dt
        dx = grid.steps[0]
        dy = grid.steps[1]
        dz = grid.steps[2]

        # Discretized partial derivatives of electric field (indexing is done to match grid.Yee_grid)
        dex_dy = (grid.ex[i, j + 1, k] - grid.ex[i, j, k]) / dy
        dex_dz = (grid.ex[i, j, k + 1] - grid.ex[i, j, k]) / dz
        dey_dx = (grid.ey[i + 1, j, k] - grid.ey[i, j, k]) / dx
        dey_dz = (grid.by[i, j, k + 1] - grid.ey[i, j, k]) / dz
        dez_dx = (grid.bz[i + 1, j, k] - grid.ez[i, j, k]) / dx
        dez_dy = (grid.bz[i, j + 1, k] - grid.ez[i, j, k]) / dy

        # Yee's scheme
        grid.bx[i, j, k] += cdt * (dey_dz - dez_dy)
        grid.by[i, j, k] += cdt * (dez_dx - dex_dz)
        grid.bz[i, j, k] += cdt * (dex_dy - dey_dx)

    def apply_b_bc(self, grid):
        """Apply periodic boundary conditions for the magnetic field"""
        self._apply_component_bc(grid.bx, grid.num_internal_cells, grid.num_guard_cells)
        self._apply_component_bc(grid.by, grid.num_internal_cells, grid.num_guard_cells)
        self._apply_component_bc(grid.bz, grid.num_internal_cells, grid.num_guard_cells)

    def _apply_component_bc(self, scalar_field, num_internal_cells, num_guard_cells):
        """Apply periodic boundary conditions for a field component"""

        num_cells = num_internal_cells + num_guard_cells * 2
        internal_start = num_guard_cells
        internal_end = num_cells - num_guard_cells

        # x axis guard: process only internal area in y, z
        for j in range(internal_start[1], internal_end[1]):
            for k in range(internal_start[2], internal_end[2]):
                for i in range (num_guard_cells[0]):
                    period = num_internal_cells[0]
                    guard_size = num_guard_cells[0]
                    # left guard: copy from right of internal area to guard
                    scalar_field[i, j, k] = scalar_field[i + period, j, k]
                    # right guard: copy from left of internal area to guard
                    scalar_field[i + guard_size + period, j, k] = scalar_field[i + guard_size, j, k]

        # y axis guard: process all in x, only internal in z
        for i in range(num_cells[0]):
            for k in range(internal_start[2], internal_end[2]):
                for j in range (num_guard_cells[1]):
                    period = num_internal_cells[1]
                    guard_size = num_guard_cells[1]
                    # left guard: copy from right of internal area to guard
                    scalar_field[i, j, k] = scalar_field[i, j + period, k]
                    # right guard: copy from left of internal area to guard
                    scalar_field[i, j + guard_size + period, k] = scalar_field[i, j + guard_size, k]

        # z axis guard: process all in x, y
        for i in range(num_cells[0]):
            for j in range(num_cells[1]):
                for k in range (num_guard_cells[2]):
                    period = num_internal_cells[2]
                    guard_size = num_guard_cells[2]
                    # left guard: copy from right of internal area to guard
                    scalar_field[i, j, k] = scalar_field[i, j, k + period]
                    # right guard: copy from left of internal area to guard
                    scalar_field[i, j, k + guard_size + period] = scalar_field[i, j, k + guard_size]
