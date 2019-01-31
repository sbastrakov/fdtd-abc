import grid.yee
import solver.fdtd
import solver.fdtdpml
import initialconditions

import math
import matplotlib.pyplot as plt
import numpy as np

def init_solver():
    # Set up solver, for now parameters are hardcoded here
    num_pml_cells = np.array([8, 8, 8])
    return solver.fdtdpml.FdtdPML(num_pml_cells)
    #return solver.fdtd.Fdtd()

def init_grid(solver):
    # Set up grid, for now parameters are hardcoded here
    min_position = np.array([0.0, 0.0, 0.0])
    max_position = np.array([1.0, 1.0, 1.0])
    num_internal_cells = np.array([64, 64, 2]) # modify this
    num_guard_cells = np.array(solver.get_guard_size()) # do not modify this
    gr = grid.yee.Yee_grid(min_position, max_position, num_internal_cells, num_guard_cells)
    # Set up initial conditions as a plane wave in Ey, Bz, runs along x in positive direction
    num_periods = 4
    initialconditions.planewave(gr, num_periods)
    return gr


def plot(grid, iteration):
    slice_z = grid.num_cells[2] // 2
    ey = np.transpose(grid.ey[:, :, slice_z])
    bz = np.transpose(grid.bz[:, :, slice_z])
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    axs[0].imshow(ey)
    axs[0].set_title('Ey(x, y, z_center)')
    axs[0].set_xlabel('x (cells)')
    axs[0].set_ylabel('y (cells)')
    fig.suptitle('Iteration ' + str(iteration), fontsize=16)
    axs[1].imshow(bz)
    axs[1].set_title('Bz(x, y, z_center)')
    axs[1].set_xlabel('x (cells)')
    axs[1].set_ylabel('y (cells)')

def main():
    solver = init_solver()
    grid = init_grid(solver)

    c = 29979245800.0 # cm / s
    # set time step to be 1% below CFL for Yee solver
    dt = 0.99 / (1.0 / grid.steps[0]**2 + 1.0 / grid.steps[1]**2 + 1.0 / grid.steps[2]**2) / c

    num_iterations = 100
    plotting_period = 10 # period to make plots, set to 0 to disable plotting

    for iteration in range(num_iterations):
        if plotting_period and iteration % plotting_period == 0:
            print("Iteration " + str(iteration))
            plot(grid, iteration)
        solver.run_iteration(grid, dt)
    if plotting_period:
        plt.show()


if __name__ == "__main__":
    main()
