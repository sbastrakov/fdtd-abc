import scipy.constants


class Printer:

    def __init__(self, period = 10, use_z_slice = False):
        self._period = period
        self._use_z_slice = use_z_slice

    def print(self, grid, iteration):
        if iteration % self._period != 0:
            return
        e_energy = 0.0
        b_energy = 0.0
        e_energy_internal = 0.0
        b_energy_internal = 0.0
        k_range = range(grid.num_cells[2])
        if self._use_z_slice:
            k_center = grid.num_cells[2] // 2
            k_range = range(k_center, k_center + 1)
        for i in range(grid.num_cells[0]):
            for j in range(grid.num_cells[1]):
                for k in k_range:
                    e_energy += grid.ex[i, j, k] * grid.ex[i, j, k] + grid.ey[i, j, k] * grid.ey[i, j, k] + grid.ez[i, j, k] * grid.ez[i, j, k]
                    b_energy += grid.bx[i, j, k] * grid.bx[i, j, k] + grid.by[i, j, k] * grid.by[i, j, k] + grid.bz[i, j, k] * grid.bz[i, j, k]
                    if self._is_internal(grid, [i, j, k]):
                        e_energy_internal += grid.ex[i, j, k] * grid.ex[i, j, k] + grid.ey[i, j, k] * grid.ey[i, j, k] + grid.ez[i, j, k] * grid.ez[i, j, k]
                        b_energy_internal += grid.bx[i, j, k] * grid.bx[i, j, k] + grid.by[i, j, k] * grid.by[i, j, k] + grid.bz[i, j, k] * grid.bz[i, j, k]
        e_factor = 0.5 * scipy.constants.epsilon_0 * grid.steps[0] * grid.steps[1] * grid.steps[2]
        e_energy *= e_factor
        e_energy_internal *= e_factor
        b_factor = 0.5 / scipy.constants.mu_0 * grid.steps[0] * grid.steps[1] * grid.steps[2]
        b_energy *= b_factor
        b_energy_internal *= b_factor
        energy = e_energy + b_energy
        energy_internal = e_energy_internal + b_energy_internal
        print(str(iteration) + " " + str(e_energy) + " " + str(b_energy) + " " + str(e_energy_internal) + " " + str(b_energy_internal))

    def _is_internal(self, grid, index):
        result = True
        ## hack for the FDTD reference setup
        ##for d in range(2):
        ##    result = result and (index[d] >= 500) and (index[d] < 540)
        for d in range(3):
            result = result and (index[d] >= grid.num_guard_cells_left[d]) and (index[d] < grid.num_cells[d] - grid.num_guard_cells_right[d])
        return result
