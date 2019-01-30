import grid.yee
import solver.fdtd
import solver.fdtdpml
import initialconditions

import numpy as np

def init_grid():
    # Set up grid, for now parameters are hardcoded here
    min_position = np.array([0.0, 0.0, 0.0])
    max_position = np.array([1.0, 1.0, 1.0])
    num_cells = np.array([64, 64, 64])
    dt = 1e-10
    gr = grid.yee.Yee_grid(dt, min_position, max_position, num_cells)

    # Set up initial conditions as a plane wave in Ey, Bz, runs along x in positive direction
    num_periods = 4
    initialconditions.planewave(gr, num_periods)

    return gr

def init_solver(grid):
    # Set up solver, for now parameters are hardcoded here
    ##return solver.fdtd.Fdtd()
    num_pml_cells = np.array([8, 8, 8])
    return solver.fdtdpml.FdtdPML(num_pml_cells, grid.steps)

def run_iteration(grid, solver, iteration):
    print("Iteration " + str(iteration))
    solver.run_iteration(grid)

def main():
    grid = init_grid()
    solver = init_solver(grid)
    num_iterations = 100
    for iteration in range(num_iterations):
        run_iteration(grid, solver, iteration)


if __name__ == "__main__":
    main()
