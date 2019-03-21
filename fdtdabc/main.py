import grid.yee
import solver.fdtd
import solver.cpml
import solver.pml_sf
import initialconditions

import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.constants

def init_solver():
    # Set up solver, for now parameters are hardcoded here
    num_pml_cells = np.array([12, 12, 4])
    #return solver.cpml.CPML(num_pml_cells, 4)
    #return solver.pml_sf.PML_SF(num_pml_cells, 4)
    return solver.fdtd.Fdtd()

def init_grid(solver):
    # Set up grid, for now parameters are hardcoded here
    min_position = np.array([0.0, 0.0, 0.0])
    max_position = np.array([1.0, 1.0, 1.0])
    num_internal_cells = np.array([64, 64, 1]) # modify this
    num_guard_cells = np.array(solver.get_guard_size(num_internal_cells)) # do not modify this
    gr = grid.yee.Yee_grid(min_position, max_position, num_internal_cells, num_guard_cells)
    # Set up initial conditions as a plane wave in Ey, Bz, runs along x in positive direction
    #num_periods = 4
    #initialconditions.planewave(gr, num_periods)
    return gr

class Plot:
    def __init__(self, period):
        self.period = period
        #self.fig, self.axs = plt.subplots(1, 2, constrained_layout=True)
        #self.axs[0].set_title('Ey(x, y, z_center)')
        #self.axs[0].set_xlabel('x (cells)')
        #self.axs[0].set_ylabel('y (cells)')
        #self.axs[1].set_title('Bz(x, y, z_center)')
        #self.axs[1].set_xlabel('x (cells)')
        #self.axs[1].set_ylabel('y (cells)')
        self.fig = plt.figure()
        self.images = []

    def add_frame(self, grid, iteration):
        if (self.period == 0) or (iteration % self.period != 0):
            return
        slice_z = grid.num_cells[2] // 2
        #ey = np.transpose(grid.ey[:, :, slice_z])
        #bz = np.transpose(grid.bz[:, :, slice_z])
        #self.fig.suptitle('Iteration ' + str(iteration), fontsize=16)
        #self.images.append(self.axs[0].imshow(ey))
        #self.axs[1].imshow(bz)
        ez = np.transpose(grid.ez[:, :, slice_z])
        self.images.append([plt.imshow(ez)])

    def animate(self):
        if self.period == 0:
            return
        ani = animation.ArtistAnimation(self.fig, self.images, interval=500, blit=True,
                                repeat_delay=1000)
        plt.show()


def add_source(grid, iteration):
    #add_soft_source(grid, iteration)
    add_hard_source(grid, iteration)

def add_soft_source(grid, iteration):
    duration_iterations = 10
    if iteration > duration_iterations:
        return
    diration_center = duration_iterations / 2
    coeff_t = math.pow(math.sin(math.pi * (iteration - diration_center) / duration_iterations), 2)
    width = [8, 8, 1]
    i_center = grid.num_cells[0] // 2
    j_center = grid.num_cells[1] // 2
    k_center = grid.num_cells[2] // 2
    for i in range(i_center - width[0] // 2, i_center + width[0] // 2 + 1):
        coeff_x = math.pow(math.cos(math.pi * (i - i_center) / width[0]), 2)
        for j in range(j_center - width[1] // 2, j_center + width[1] // 2 + 1):
            coeff_y = math.pow(math.cos(math.pi * (j - j_center) / width[1]), 2)
            for k in range(k_center - width[2] // 2, k_center + width[2] // 2 + 1):
                coeff_z = math.pow(math.cos(math.pi * (k - k_center) / width[2]), 2)
                grid.ez[i, j, k] += coeff_x * coeff_y * coeff_z * coeff_t

def add_hard_source(grid, iteration):
    """(6.46) from Taflove 2nd ed."""
    duration_iterations = 40
    duration_center = duration_iterations / 2
    i_center = grid.num_cells[0] // 2
    j_center = grid.num_cells[1] // 2
    k_center = grid.num_cells[2] // 2
    value = (10.0 - 15 * math.cos(math.pi * iteration / duration_center) + 6 * math.cos(2.0 * math.pi * iteration / duration_center) - math.cos(3.0 * math.pi * iteration / duration_center)) / 32.0
    if iteration > duration_iterations:
        value = 0.0
    grid.ez[i_center, j_center, k_center] = value
    #for j in range(grid.num_cells[1]):
    #    grid.ez[i_center, j, k_center] = value

def print_energy(grid, iteration):
    period = 20
    if iteration % period != 0:
        return
    e_energy = 0.0
    b_energy = 0.0
    e_energy_internal = 0.0
    b_energy_internal = 0.0
    for i in range(grid.num_cells[0]):
        for j in range(grid.num_cells[1]):
            # for k in range(grid.num_cells[2]):
            #     node_value = grid.ex[i, j, k] * grid.ex[i, j, k] + grid.ey[i, j, k] * grid.ey[i, j, k] + grid.ez[i, j, k] * grid.ez[i, j, k]
            #     node_value += grid.bx[i, j, k] * grid.bx[i, j, k] + grid.by[i, j, k] * grid.by[i, j, k] + grid.bz[i, j, k] * grid.bz[i, j, k]
            #     energy += node_value * grid.steps[0] * grid.steps[1] * grid.steps[2]
            k = grid.num_cells[2] // 2
            e_energy += grid.ex[i, j, k] * grid.ex[i, j, k] + grid.ey[i, j, k] * grid.ey[i, j, k] + grid.ez[i, j, k] * grid.ez[i, j, k]
            b_energy += grid.bx[i, j, k] * grid.bx[i, j, k] + grid.by[i, j, k] * grid.by[i, j, k] + grid.bz[i, j, k] * grid.bz[i, j, k]
            if (i >= grid.num_guard_cells[0]) and (i < grid.num_cells[0] - grid.num_guard_cells[0]) and (j >= grid.num_guard_cells[1]) and (j < grid.num_cells[1] - grid.num_guard_cells[1]):
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

def main():
    solver = init_solver()
    grid = init_grid(solver)

    # set time step to be 1% below CFL for Yee solver
    dt_cfl_limit = 1.0 / (scipy.constants.c * math.sqrt(1.0 / grid.steps[0]**2 + 1.0 / grid.steps[1]**2 + 1.0 / grid.steps[2]**2) )
    ratio_of_cfl_limit = 0.99
    dt = ratio_of_cfl_limit * dt_cfl_limit

    num_iterations = 200
    plotting_period = 20 # period to make plots, set to 0 to disable plotting

    plot = Plot(plotting_period)
    for iteration in range(num_iterations):
        print_energy(grid, iteration)
        plot.add_frame(grid, iteration)
        add_source(grid, iteration)
        solver.run_iteration(grid, dt)
    plot.animate()


if __name__ == "__main__":
    main()
