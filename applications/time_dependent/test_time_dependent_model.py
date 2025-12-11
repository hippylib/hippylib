# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2022, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
# Copyright (c) 2023-2024, The University of Texas at Austin 
# & University of California--Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

# This script runs a finite difference check for time-dependent PDE variational problems
# using parabolic PDEs as an example. It demonstrates the difference between 
# the TimeDependentPDEVariationalProblem class which implements derivatives for general one-step time stepping schemes
# and the ImplicitEulerTimeDependentPDEVariationalProblem class, which only handles the implicit Euler method.
# Note that the finite difference check should pass for both classes when `theta` is set to 1.0 (implicit Euler method)
# in the theta method, while the ImplicitEulerTimeDependentPDEVariationalProblem class will fail for other values of `theta`
# and for operator splitting schemes.


import argparse
import dolfin as dl
dl.set_log_active(False)

import ufl
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import STATE, ADJOINT, PARAMETER, ADJOINT, \
    TimeDependentPDEVariationalProblem, ImplicitEulerTimeDependentPDEVariationalProblem, \
    ContinuousStateObservation, MisfitTD, parRandom, BiLaplacianPrior, \
    Model, modelVerify



class LinearHeatEquationThetaMethodVarf:
    """
    Variational form for the heat equation with sinusoidal forcing term using theta method time stepping 
    """
    def __init__(self, dt : float, theta : float):
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        self.theta = theta 

        self.f = dl.Expression("4 * (sin(k * t) + 1) * (1 - x[0]) * x[0]", k=2.*np.pi, t=0., degree=2)
        self.f_old = dl.Expression("4 * (sin(k * t) + 1) * (1 - x[0]) * x[0]", k=2.*np.pi, t=0., degree=2)
        print("Testing linear heat equation")

    @property
    def dt(self):
        return self._dt
        
    def __call__(self,u,u_old, m, p, t):
        self.f.t = t 
        self.f_old.t = t - self.dt

        varf = (u - u_old)*p*self.dt_inv*ufl.dx
        varf += dl.Constant(self.theta) * ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx
        varf += dl.Constant(1-self.theta) * ufl.exp(m)*ufl.inner(ufl.grad(u_old), ufl.grad(p))*ufl.dx
        varf -= dl.Constant(self.theta) * self.f * p*ufl.dx
        varf -= dl.Constant(1-self.theta) * self.f_old * p*ufl.dx
        return varf


class NonlinearHeatEquationOneStepMethodVarf:
    """
    Variational form for the nonlinear heat equation using one-step time stepping method.
    Options are theta method or operator splitting.
    """
    def __init__(self, dt : float, theta : float, splitting: bool=False):
        self._dt = dt
        self.dt_inv = dl.Constant(1./dt)
        self.theta = theta 
        self.splitting = splitting 

        self.f = dl.Expression("4 * (sin(k * t) + 1) * (1 - x[0]) * x[0]", k=2.*np.pi, t=0., degree=2)
        self.f_old = dl.Expression("4 * (sin(k * t) + 1) * (1 - x[0]) * x[0]", k=2.*np.pi, t=0., degree=2)
        self.small_conductivity = dl.Constant(0.01)
        print("Testing nonlinear heat equation")

    @property
    def dt(self):
        return self._dt
        
    def __call__(self,u,u_old, m, p, t):
        if self.splitting:
            return self._splitting_varf(u,u_old, m, p, t)
        else:
            return self._theta_method_varf(u,u_old, m, p, t)

    def _theta_method_varf(self,u,u_old, m, p, t):
        self.f.t = t 
        self.f_old.t = t - self.dt

        varf = (u - u_old)*p*self.dt_inv*ufl.dx
        varf += dl.Constant(self.theta) * (self.small_conductivity + ufl.sin(u)**2) * ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx
        varf += dl.Constant(1-self.theta) * (self.small_conductivity + ufl.sin(u_old)**2) * ufl.exp(m)*ufl.inner(ufl.grad(u_old), ufl.grad(p))*ufl.dx
        varf -= dl.Constant(self.theta) * self.f * p*ufl.dx
        varf -= dl.Constant(1-self.theta) * self.f_old * p*ufl.dx
        return varf

    def _splitting_varf(self,u,u_old, m, p, t):
        self.f.t = t 

        varf = (u - u_old)*p*self.dt_inv*ufl.dx
        varf += (self.small_conductivity + (ufl.sin(u_old))*(ufl.sin(u))) * ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx
        varf -= self.f * p*ufl.dx
        return varf



def u_boundary(x, on_boundary):
    """Boundary marker for the top and bottom walls (y-direction).
    """
    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)


def v_boundary(x, on_boundary):
    """Boundary marker for left and right walls (x-direction).
    """
    return on_boundary and ( x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)


def make_mesh_and_function_spaces(nx : float, ny : float):
    """
    Create mesh and function spaces.
    """
    mesh = dl.UnitSquareMesh(nx, ny)
    Vh_STATE = dl.FunctionSpace(mesh, "CG", 2)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE]
    return Vh


def make_heat_equation_pde_problem(Vh : list[dl.FunctionSpace],
                                   t_final : float, 
                                   nt : int, 
                                   theta : float, 
                                   is_nonlinear : bool, 
                                   is_splitting : bool, 
                                   one_step_pde_class : bool):
    """
    Create a time-dependent PDE variational problem for the heat equation.
    """

    if is_splitting:
        assert is_nonlinear, "Operator splitting is only implemented for the nonlinear heat equation."

    t_init = 0 
    dt = (t_final - t_init)/nt

    # Boundary condition
    u_bdr = dl.Expression("x[1]", element=Vh[STATE].ufl_element())
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)
    u0_expr = dl.Expression("x[1] + x[0]*(1.-x[0])*x[1]*(1.-x[1])", element=Vh[STATE].ufl_element())
    u0 = dl.interpolate(u0_expr, Vh[STATE])

    if is_nonlinear:
        is_fwd_linear = False
        varf = NonlinearHeatEquationOneStepMethodVarf(dt, theta, splitting=is_splitting)
    else:
        is_fwd_linear = True
        varf = LinearHeatEquationThetaMethodVarf(dt, theta)

    if one_step_pde_class:
        pde_problem = TimeDependentPDEVariationalProblem(Vh, varf, bc, bc0, u0, t_init, t_final, is_fwd_linear=is_fwd_linear)
    else:
        pde_problem = ImplicitEulerTimeDependentPDEVariationalProblem(Vh, varf, bc, bc0, u0, t_init, t_final, is_fwd_linear=is_fwd_linear)

    return pde_problem 


def make_prior(Vh):
    """
    Create a Gaussian prior with Bi-Laplacian covariance operator.
    """
    GAMMA = 0.1 
    DELTA = 1.0 
    prior = BiLaplacianPrior(Vh[PARAMETER], gamma=GAMMA, delta=DELTA)
    return prior 


def make_continuous_observation_misfit(Vh, pde):
    """
    Create a time-dependent misfit object with continuous state observations at all time steps.
    """
    TARGET_EXPR = dl.Expression("cos(k * t) * x[0] * x[1]", k=2.*np.pi, t=0., degree=2)
    bc0 = pde.adj_bc

    # Set up misfit object.
    misfits = []
    for t in pde.times:
        misfit_t = ContinuousStateObservation(Vh[STATE], ufl.dx, bc0)

        TARGET_EXPR.t = t
        target_fun = dl.interpolate(TARGET_EXPR, Vh[STATE])
        misfit_t.d.axpy(1.0, target_fun.vector())
        misfit_t.noise_variance = 1.0 
        misfits.append(misfit_t)
    
    misfit = MisfitTD(misfits, pde.times)
    return misfit 


def sample_from_prior(prior):
    """
    Sample from the prior distribution.
    """
    m_sample = dl.Function(prior.Vh)
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    parRandom.normal(1.0, noise)
    prior.sample(noise, m_sample.vector())
    return m_sample


def make_problem_name(theta, is_nonlinear, is_splitting, one_step_pde_class):
    """
    Create a filename for exporting results.
    """

    problem_name = f"heat_equation"

    if is_nonlinear:
        problem_name += f"_nonlinear"
    else:
        problem_name += f"_linear"

    if is_splitting:
        problem_name += f"_splitting"
    else:
        problem_name += f"_theta{theta}"
    
    if one_step_pde_class:
        problem_name += f"_one_step_class"
    else:
        problem_name += f"_implicit_euler_class"

    return problem_name

def make_problem_title(theta, is_nonlinear, is_splitting, one_step_pde_class):
    """
    Create a title for plots.
    """

    problem_title = f"Heat Equation"

    if is_nonlinear:
        problem_title += f" (Nonlinear)"
    else:
        problem_title += f" (Linear)"

    if is_splitting:
        problem_title += f" with Operator Splitting"
    else:
        problem_title += f" with Theta Method (theta={theta})"
    
    if one_step_pde_class:
        problem_title += f"\n[TimeDependentPDEVariationalProblem]"
    else:
        problem_title += f"\n[ImplicitEulerTimeDependentPDEVariationalProblem]"

    return problem_title



def run_heat_equation_comparison(nx : int,
                                 ny : int, 
                                 t_final : float, 
                                 nt : int, 
                                 theta : float, 
                                 is_nonlinear : bool, 
                                 is_splitting : bool,
                                 no_plots : bool = True):

    """
    Run a finite difference check time-dependent PDE variational problem for the heat equation.
    Compare results using the TimeDependentPDEVariationalProblem class
    and the ImplicitEulerTimeDependentPDEVariationalProblem class.

    :param nx: Number of elements in x-direction.
    :param ny: Number of elements in y-direction.
    :param t_final: Final time.
    :param nt: Number of time steps.
    :param theta: Theta parameter for theta method (0 <= theta <= 1). Not
                    used if using operator splitting.
    :param is_nonlinear: Use nonlinear heat equation.
    :param is_splitting: Use operator splitting for nonlinear heat equation. Only used if is_nonlinear is set.
    :param no_plots: Do not show plots.
    """
    PLOT_DIR = "test_plots"
    SOLUTION_DIR = "test_solutions"
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(SOLUTION_DIR, exist_ok=True)

    Vh = make_mesh_and_function_spaces(nx, ny)

    prior = make_prior(Vh)

    implicit_euler_pde_problem = make_heat_equation_pde_problem(Vh, 
                                                          t_final, 
                                                          nt, 
                                                          theta, 
                                                          is_nonlinear, 
                                                          is_splitting, 
                                                          one_step_pde_class=False)

    implicit_euler_misfit = make_continuous_observation_misfit(Vh, implicit_euler_pde_problem)

    implicit_euler_problem_name = make_problem_name(theta, 
                                              is_nonlinear, 
                                              is_splitting, 
                                              one_step_pde_class=False)

    implicit_euler_plot_title = make_problem_title(theta, 
                                             is_nonlinear, 
                                             is_splitting, 
                                             one_step_pde_class=False)

    implicit_euler_model = Model(implicit_euler_pde_problem, prior, implicit_euler_misfit)


    one_step_pde_problem = make_heat_equation_pde_problem(Vh, 
                                                          t_final, 
                                                          nt, 
                                                          theta, 
                                                          is_nonlinear, 
                                                          is_splitting, 
                                                          one_step_pde_class=True)


    one_step_misfit = make_continuous_observation_misfit(Vh, one_step_pde_problem)
    one_step_model = Model(one_step_pde_problem, prior, one_step_misfit)

    one_step_problem_name = make_problem_name(theta, 
                                              is_nonlinear, 
                                              is_splitting, 
                                              one_step_pde_class=True)

    one_step_plot_title = make_problem_title(theta, 
                                             is_nonlinear, 
                                             is_splitting, 
                                             one_step_pde_class=True)

    m0 = sample_from_prior(prior)

    x = implicit_euler_model.generate_vector()
    x[PARAMETER].axpy(1.0, m0.vector())

    SEP = "-"*80

    # -------------------------------------------------------------- # 
    # Solve forward problem and FD check for IMPLICIT-EULER PDE class 
    # -------------------------------------------------------------- # 
    print(SEP)
    print("Check using ImplicitEulerTimeDependentPDEVariationalProblem class:")
    print(SEP)
    print("Solving forward problem using ImplicitEuler PDE class...")
    implicit_euler_model.solveFwd(x[STATE], x)
    implicit_euler_pde_problem.exportState(x[STATE], f"{SOLUTION_DIR}/{implicit_euler_problem_name}_state.xdmf")

    print("Finite difference check for ImplicitEuler PDE class...")
    modelVerify(implicit_euler_model, m0.vector(), misfit_only=True)
    fig = plt.gcf()
    fig.suptitle(implicit_euler_plot_title)
    fig.set_size_inches(12,6)
    plt.savefig(f"{PLOT_DIR}/{implicit_euler_problem_name}_fdcheck.png")

    print(SEP, "\n")

    # -------------------------------------------------------------- # 
    # Solve forward problem and FD check for ONE-STEP PDE class 
    # -------------------------------------------------------------- # 

    x = one_step_model.generate_vector()
    x[PARAMETER].axpy(1.0, m0.vector())

    print(SEP)
    print("Check using one-step TimeDependentPDEVariationalProblem class:")
    print(SEP)
    print("Solving forward problem using one-step PDE class...")
    one_step_model.solveFwd(x[STATE], x)
    one_step_pde_problem.exportState(x[STATE], f"{SOLUTION_DIR}/{one_step_problem_name}_state.xdmf")

    print("Finite difference check for one-step PDE class...")
    modelVerify(one_step_model, m0.vector(), misfit_only=True)
    fig = plt.gcf()
    fig.suptitle(one_step_plot_title)
    fig.set_size_inches(12,6)
    plt.savefig(f"{PLOT_DIR}/{one_step_problem_name}_fdcheck.png")

    print(SEP, "\n")
    if not no_plots:
        plt.show()


def main(args):

    print(args)

    nx = args.nx
    ny = args.ny
    t_final = args.tfinal
    nt = args.nt
    theta = args.theta
    is_nonlinear = args.nonlinear
    is_splitting = args.splitting
    no_plots = args.no_plots

    run_heat_equation_comparison(nx, ny, t_final, nt, theta, is_nonlinear, is_splitting, no_plots)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test time-dependent heat equation model.")
    parser.add_argument("--nx", type=int, default=16, 
                        help="Number of elements in x-direction.")

    parser.add_argument("--ny", type=int, default=16, 
                        help="Number of elements in y-direction.")

    parser.add_argument("--tfinal", type=float, default=0.1, 
                        help="Final time.")

    parser.add_argument("--nt", type=int, default=10, 
                        help="Number of time steps.")

    parser.add_argument("--theta", type=float, default=0.5, 
                        help="Theta parameter for theta method (0 <= theta <= 1). " \
                        "Not used if using operator splitting.")

    parser.add_argument("--nonlinear", default=False, action="store_true", 
                        help="Use nonlinear heat equation.")

    parser.add_argument("--splitting", default=False, action="store_true", 
                        help="Use operator splitting for nonlinear heat equation. Only used if --nonlinear is set.")

    parser.add_argument("--no_plots", default=False, action="store_true", 
                        help="Do not show plots.")

    args = parser.parse_args()

    main(args)
