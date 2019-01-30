import numpy as np
import math

class Initial_conditions:

    def __init__(self, ex = None, ey = None, ez = None, bx = None, by = None, bz = None):
        self.ex = ex
        self.ey = ey
        self.ez = ez
        self.bx = bx
        self.by = by
        self.bz = bz

def planewave(grid, num_periods):
    """Plane wave in ey, bz propagating along x"""
    wavelength = (grid.max_position[0] - grid.min_position[1]) / num_periods
    frequency = 2.0 * math.pi / wavelength 
    wave = lambda position : math.sin((position[0] - grid.min_position[0]) * frequency)
    ic = Initial_conditions(ey = wave, bz = wave)
    apply(ic, grid)

def apply(initial_conditions, grid):
    _apply_component(initial_conditions.ex, grid.ex)
    _apply_component(initial_conditions.ey, grid.ey)
    _apply_component(initial_conditions.ez, grid.ez)        
    _apply_component(initial_conditions.bx, grid.bx)
    _apply_component(initial_conditions.by, grid.by)
    _apply_component(initial_conditions.bz, grid.bz)   

def _apply_component(function, component):
    if function is None:
        return
    for i in range(component.num_cells[0]):
        for j in range(component.num_cells[1]):
            for k in range(component.num_cells[2]):
                position = component.position([i, j, k])
                component[i, j, k] = function(position)
