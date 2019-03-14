import grid.yee
import solver.fdtd
import solver.fdtdpml
import initialconditions

import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def init_solver():
    # Set up solver, for now parameters are hardcoded here
    num_pml_cells = np.array([8, 8, 0])
    return solver.fdtdpml.FdtdPML(num_pml_cells, 2)
    #return solver.fdtd.Fdtd()

def init_grid(solver):
    # Set up grid, for now parameters are hardcoded here
    min_position = np.array([0.0, 0.0, 0.0])
    max_position = np.array([1.0, 1.0, 1.0])
    num_internal_cells = np.array([32, 32, 3]) # modify this
    num_guard_cells = np.array(solver.get_guard_size()) # do not modify this
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
    duration_iterations = 10
    if iteration > duration_iterations:
        return
    i = grid.num_cells[0] // 2
    j = grid.num_cells[1] // 2
    k = grid.num_cells[2] // 2
    sin_arg = math.sin(2.0 * math.pi * float(iteration) / float(duration_iterations))
    grid.ez[i, j, k] += math.pow(math.sin(sin_arg), 2.0)

def print_energy(grid, iteration):
    period = 10
    if iteration % period != 0:
        return
    energy = 0.0
    for i in range(grid.num_cells[0]):
        for j in range(grid.num_cells[1]):
            for k in range(grid.num_cells[2]):
                node_value = grid.ex[i, j, k] * grid.ex[i, j, k] + grid.ey[i, j, k] * grid.ey[i, j, k] + grid.ez[i, j, k] * grid.ez[i, j, k]
                node_value += grid.bx[i, j, k] * grid.bx[i, j, k] + grid.by[i, j, k] * grid.by[i, j, k] + grid.bz[i, j, k] * grid.bz[i, j, k]
                energy += node_value * grid.steps[0] * grid.steps[1] * grid.steps[2]
    energy *= 1e-7 / (8.0 * math.pi)
    print(str(iteration) + " " + str(energy))

def main():
    solver = init_solver()
    grid = init_grid(solver)

    c = 29979245800.0 # cm / s

    # set time step to be 1% below CFL for Yee solver
    dt_cfl_limit = 1.0 / (c * math.sqrt(1.0 / grid.steps[0]**2 + 1.0 / grid.steps[1]**2 + 1.0 / grid.steps[2]**2) )
    dt = 0.99 * dt_cfl_limit

    num_iterations = 100
    plotting_period = 10 # period to make plots, set to 0 to disable plotting

    plot = Plot(plotting_period)
    for iteration in range(num_iterations):
        print_energy(grid, iteration)
        plot.add_frame(grid, iteration)
        add_source(grid, iteration)
        solver.run_iteration(grid, dt)
    plot.animate()


if __name__ == "__main__":
    main()
