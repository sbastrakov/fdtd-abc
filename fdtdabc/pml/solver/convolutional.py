import numpy as np
import scipy.constants

from pml.parameters.polymonial_grading import Parameter
from pml.parameters.sigma_max import get_sigma_max
from pml.solver.base import Solver as SolverBase

import math


class Solver(SolverBase):
    """
    Yee's FDTD solver with convolutional PML
    Operates on Yee grid
    All units in SI
    Implementation and notation are based on
    Branko D. Gvozdic, Dusan Z. Djurdjevic. Performance advantages of CPML over UPML absorbing boundary conditions in FDTD algorithm.
    Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53
     https://www.degruyter.com/downloadpdf/j/jee.2017.68.issue-1/jee-2017-0006/jee-2017-0006.pdf    
    """    

    def __init__(self, num_pml_cells_left, num_pml_cells_right, sigma_kappa_order = 4, kappa_max = np.array([15.0, 15.0, 15.0]), alpha_order = 1.0, alpha_max = np.array([0.24, 0.24, 0.24])):
        SolverBase.__init__(self, num_pml_cells_left, num_pml_cells_right)
        self._sigma_kappa_order = sigma_kappa_order
        self._kappa_max = kappa_max
        self._alpha_order = alpha_order
        self._alpha_max = alpha_max

    def _init_coeffs(self, grid, dt):
        sigma_min = np.zeros(3)
        sigma_max = get_sigma_max(grid.steps, self._sigma_kappa_order)
        self._sigma = Parameter(grid, self.num_pml_cells_left, self.num_pml_cells_right, sigma_min, sigma_max, self._sigma_kappa_order, True)
        self._is_internal = Parameter(grid, self.num_pml_cells_left, self.num_pml_cells_right, np.zeros(3), np.ones(3), 1, True)
        kappa_min = np.ones(3)
        self._kappa = Parameter(grid, self.num_pml_cells_left, self.num_pml_cells_right, kappa_min, self._kappa_max, self._sigma_kappa_order, True)
        alpha_min = np.zeros(3)
        self._alpha = Parameter(grid, self.num_pml_cells_left, self.num_pml_cells_right, self._alpha_max, alpha_min, self._alpha_order, False)
        SolverBase._init_coeffs(self, grid, dt)

    def _compute_coeff(self, coeffs, i, j, k, shift, dt):
        eps0 = scipy.constants.epsilon_0
        index = np.array([i, j, k]) + shift
        sigma = self._sigma.get(index)
        kappa = self._kappa.get(index)
        alpha = self._alpha.get(index)
        psi_b = [1.0, 1.0, 1.0]
        psi_c = [0.0, 0.0, 0.0]
        coeffs.is_internal[i, j, k] = np.array_equal(self._is_internal.get(index), np.zeros(3))
        for d in range(3):
            psi_b[d] = math.exp(-(sigma[d] / kappa[d] + alpha[d]) * dt / eps0)
            if sigma[d] + alpha[d] * kappa[d] != 0.0:
                psi_c[d] = sigma[d] * (psi_b[d] - 1.0) / (kappa[d] * (sigma[d] + alpha[d] * kappa[d]))
        coeffs.decay_x[i, j, k] = psi_b[0]
        coeffs.decay_y[i, j, k] = psi_b[1]
        coeffs.decay_z[i, j, k] = psi_b[2]
        coeffs.diff_x[i, j, k] = psi_c[0]
        coeffs.diff_y[i, j, k] = psi_c[1]
        coeffs.diff_z[i, j, k] = psi_c[2]

    def _update_pml_e_element(self, grid, dt, i, j, k, dbx_dy, dbx_dz, dby_dx, dby_dz, dbz_dx, dbz_dy):
        index = np.array([i, j, k])
        kappa = self._kappa.get(index)
        coeffs = self._e_coeffs
        psi_e = self._e_split
        psi_e.yx[i, j, k] = coeffs.decay_x[i, j, k] * psi_e.yx[i, j, k] + coeffs.diff_x[i, j, k] * dbz_dx
        psi_e.zx[i, j, k] = coeffs.decay_x[i, j, k] * psi_e.zx[i, j, k] + coeffs.diff_x[i, j, k] * dby_dx
        psi_e.xy[i, j, k] = coeffs.decay_y[i, j, k] * psi_e.xy[i, j, k] + coeffs.diff_y[i, j, k] * dbz_dy
        psi_e.zy[i, j, k] = coeffs.decay_y[i, j, k] * psi_e.zy[i, j, k] + coeffs.diff_y[i, j, k] * dbx_dy
        psi_e.xz[i, j, k] = coeffs.decay_z[i, j, k] * psi_e.xz[i, j, k] + coeffs.diff_z[i, j, k] * dby_dz
        psi_e.yz[i, j, k] = coeffs.decay_z[i, j, k] * psi_e.yz[i, j, k] + coeffs.diff_z[i, j, k] * dbx_dz
        coeff = dt / (scipy.constants.epsilon_0 * scipy.constants.mu_0)
        grid.ex[i, j, k] += coeff * (dbz_dy / kappa[1] - dby_dz / kappa[2] + psi_e.xy[i, j, k] - psi_e.xz[i, j, k])
        grid.ey[i, j, k] += coeff * (dbx_dz / kappa[2] - dbz_dx / kappa[0] + psi_e.yz[i, j, k] - psi_e.yx[i, j, k])
        grid.ez[i, j, k] += coeff * (dby_dx / kappa[0] - dbx_dy / kappa[1] + psi_e.zx[i, j, k] - psi_e.zy[i, j, k])

    def _update_pml_b_element(self, grid, dt, i, j, k, dex_dy, dex_dz, dey_dx, dey_dz, dez_dx, dez_dy):
        index = np.array([i + 0.5, j + 0.5, k + 0.5])
        kappa = self._kappa.get(index)
        coeffs = self._b_coeffs
        psi_b = self._b_split
        psi_b.yx[i, j, k] = coeffs.decay_x[i, j, k] * psi_b.yx[i, j, k] + coeffs.diff_x[i, j, k] * dez_dx
        psi_b.zx[i, j, k] = coeffs.decay_x[i, j, k] * psi_b.zx[i, j, k] + coeffs.diff_x[i, j, k] * dey_dx
        psi_b.xy[i, j, k] = coeffs.decay_y[i, j, k] * psi_b.xy[i, j, k] + coeffs.diff_y[i, j, k] * dez_dy
        psi_b.zy[i, j, k] = coeffs.decay_y[i, j, k] * psi_b.zy[i, j, k] + coeffs.diff_y[i, j, k] * dex_dy
        psi_b.xz[i, j, k] = coeffs.decay_z[i, j, k] * psi_b.xz[i, j, k] + coeffs.diff_z[i, j, k] * dey_dz
        psi_b.yz[i, j, k] = coeffs.decay_z[i, j, k] * psi_b.yz[i, j, k] + coeffs.diff_z[i, j, k] * dex_dz
        grid.bx[i, j, k] += dt * (dey_dz / kappa[2] - dez_dy / kappa[1] + psi_b.xz[i, j, k] - psi_b.xy[i, j, k])
        grid.by[i, j, k] += dt * (dez_dx / kappa[0] - dex_dz / kappa[2] + psi_b.yx[i, j, k] - psi_b.yz[i, j, k])
        grid.bz[i, j, k] += dt * (dex_dy / kappa[1] - dey_dx / kappa[0] + psi_b.zy[i, j, k] - psi_b.zx[i, j, k])
