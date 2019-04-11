import fdtd.solver as fdtd
from grid.yee import YeeGrid as Grid
from output.energy import Printer as EnergyPrinter
from output.plot import Plot
import pml.solver.convolutional as cpml
import pml.solver.split_field as sfpml

import math
import numpy as np
import scipy.constants


def init_solver():
    # Set up solver, for now parameters are hardcoded here
    pml_left_width_cells = 10
    pml_right_width_cells = 10
    num_pml_cells_left = np.array([pml_left_width_cells, pml_left_width_cells, pml_left_width_cells])
    num_pml_cells_right = np.array([pml_right_width_cells, pml_right_width_cells, pml_right_width_cells])
    return cpml.Solver(num_pml_cells_left, num_pml_cells_right, 3, 1.0, np.array([1.0, 1.0, 1.0]), 1.0, np.array([0.2, 0.2, 0.2]))
    return sfpml.Solver(num_pml_cells_left, num_pml_cells_right, 3, True)
    return fdtd.Solver()

def init_grid(solver):
    # Set up grid, for now parameters are hardcoded here
    num_internal_cells = np.array([40, 40, 1]) #np.array([1040, 1040, 1]))
    steps = np.array([1e-3, 1e-3, 1e-3])
    size = steps * num_internal_cells
    min_position = size * -0.5
    max_position = size * 0.5
    num_guard_cells_left, num_guard_cells_right = np.array(solver.get_guard_size(num_internal_cells)) # do not modify this
    grid = Grid(min_position, max_position, num_internal_cells, num_guard_cells_left, num_guard_cells_right)
    return grid


def add_source(grid, iteration, dt):
    add_soft_gaussian_source(grid, iteration, dt)
    #add_soft_source(grid, iteration)
    #add_hard_source(grid, iteration)

def add_soft_gaussian_source(grid, iteration, dt):
    # Source parameters in Taflove 3rd ed., (7.134)
    gw = 26.53e-12 # e-12 is ps
    td = 4 * gw
    t = iteration * dt
    normalized_t = (t - td) / gw
    value = -2.0 * normalized_t * math.exp(-normalized_t * normalized_t)
    i_center = grid.num_cells[0] // 2
    j_center = grid.num_cells[1] // 2
    k_center = grid.num_cells[2] // 2
    grid.ey[i_center, j_center, k_center] += value * dt / scipy.constants.epsilon_0

def add_soft_source(grid, iteration):
    duration_iterations = 10
    if iteration > duration_iterations:
        return
    duration_center = duration_iterations / 2
    coeff_t = math.pow(math.sin(math.pi * (iteration - duration_center) / duration_iterations), 2)
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


def main():
    solver = init_solver()
    grid = init_grid(solver)

    # set time step to be 1% below CFL for Yee solver
    dt_cfl_limit = 1.0 / (scipy.constants.c * math.sqrt(1.0 / grid.steps[0]**2 + 1.0 / grid.steps[1]**2 + 1.0 / grid.steps[2]**2) )
    ratio_of_cfl_limit = 0.99
    dt = ratio_of_cfl_limit * dt_cfl_limit

    # Setup from Taflove 3rd ed. section 7.11.1
    num_iterations = 1000
    output_period = 10 # period to make plots, set to 0 to disable plotting

    source_idx = grid.num_cells // 2
    source_position = grid.ey.position(source_idx)
    print("source: idx = " + str(source_idx) + ", pos = " + str(source_position))
    point_a_idx = [grid.num_guard_cells_left[0] + 2, source_idx[1], source_idx[2]]
    ##point_a_idx = [502, source_idx[1], source_idx[2]]
    point_a_position = grid.ey.position(point_a_idx)
    print("point a: idx = " + str(point_a_idx) + ", pos = " + str(point_a_position))
    point_b_idx = [grid.num_guard_cells_left[0] + 2, grid.num_guard_cells_left[1] + 2, source_idx[2]]
    ##point_b_idx = [502, 502, source_idx[2]]
    point_b_position = grid.ey.position(point_b_idx)
    print("point b: idx = " + str(point_b_idx) + ", pos = " + str(point_b_position))
    ey_point_a = []
    ey_point_b = []

    energy_printer = EnergyPrinter(output_period)
    plot = Plot(output_period)
    for iteration in range(num_iterations):
        if iteration % 50 == 0:
            print("iteration = " + str(iteration))
            with open("field_a_fdtd.txt", "a") as output:
                output.write(str(ey_point_a) + "\n")
            with open("field_b_fdtd.txt", "a") as output:
                output.write(str(ey_point_b) + "\n")
        ey_point_a.append(grid.ey[point_a_idx[0], point_a_idx[1], point_a_idx[2]])
        ey_point_b.append(grid.ey[point_b_idx[0], point_b_idx[1], point_b_idx[2]])
        energy_printer.print(grid, iteration)
        plot.add_frame(grid, iteration)
        add_source(grid, iteration, dt)
        solver.run_iteration(grid, dt)
    plot.animate()


if __name__ == "__main__":
    main()
