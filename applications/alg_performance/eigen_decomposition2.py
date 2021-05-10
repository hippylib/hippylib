'''
Created on Jun 3, 2020

@author: uvilla
'''

import dolfin as dl
import ufl
import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *



def singlePassG_original(A, B, Binv, Omega,k):
    """
    The single pass algorithm for the Generalized Hermitian Eigenvalues Problems (GHEP) as presented in [2].
    
    Inputs:

    - :code:`A`: the operator for which we need to estimate the dominant generalized eigenpairs.
    - :code:`B`: the right-hand side operator.
    - :code:`Binv`: the inverse of the right-hand side operator.
    - :code:`Omega`: a random gassian matrix with :math:`m \\geq k` columns.
    - :code:`k`: the number of eigenpairs to extract.
    - :code:`s`: the number of power iterations for selecting the subspace.
    
    Outputs:

    - :code:`d`: the estimate of the :math:`k` dominant eigenvalues of :math:`A`.
    - :code:`U`: the estimate of the :math:`k` dominant eigenvectors of :math:`A,\\, U^T B U = I_k`.
    """
    nvec  = Omega.nvec()
    
    assert(nvec >= k )
    
    Ybar = MultiVector(Omega[0], nvec)
    Q = MultiVector(Omega)
    MatMvMult(A, Omega, Ybar)
    MatMvMult(Solver2Operator(Binv), Ybar, Q)
    
    BQ, _ = Q.Borthogonalize(B)
    
    X = Omega.dot_mv(BQ)
    W = Omega.dot_mv(Ybar)
    XTinvW = np.linalg.solve(np.transpose(X),W)
    T = np.linalg.solve(X, np.transpose(XTinvW))
                        
    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()
        
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
        
    U = MultiVector(Omega[0], k)
    MvDSmatMult(Q, V, U)
    
        
    return d, U

def u_boundary(x, on_boundary):
    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def v_boundary(x, on_boundary):
    return on_boundary and ( x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

def true_model(prior):
    noise = dl.Vector()
    prior.init_vector(noise,"noise")
    parRandom.normal(1., noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise,mtrue)
    return mtrue
            
def getHessian(nx, ny):

    try:
        dl.set_log_active(False)
    except:
        pass
    sep = "\n"+"#"*80+"\n"
    ndim = 2

    mesh = dl.UnitSquareMesh(nx, ny)
    
    rank = dl.MPI.rank(mesh.mpi_comm())
    nproc = dl.MPI.size(mesh.mpi_comm())
            
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    
    ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
    if rank == 0:
        print (sep, "Set up the mesh and finite element spaces", sep)
        print ("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs) )
    
    # Initialize Expressions
    f = dl.Constant(0.0)
        
    u_bdr = dl.Expression("x[1]", element = Vh[STATE].ufl_element() )
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)
    
    def pde_varf(u,m,p):
        return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx
    
    pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    pde.solver = PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
    pde.solver_fwd_inc = PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
    pde.solver_adj_inc = PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
       
    pde.solver.parameters["relative_tolerance"] = 1e-15
    pde.solver.parameters["absolute_tolerance"] = 1e-20
    pde.solver_fwd_inc.parameters = pde.solver.parameters
    pde.solver_adj_inc.parameters = pde.solver.parameters
 
    ntargets = 50
    np.random.seed(seed=1)
    #Targets only on the bottom
    targets_x = np.random.uniform(0.1,0.9, [ntargets] )
    targets_y = np.random.uniform(0.1,0.5, [ntargets] )
    targets = np.zeros([ntargets, ndim])
    targets[:,0] = targets_x
    targets[:,1] = targets_y
    #targets everywhere
    #targets = np.random.uniform(0.1,0.9, [ntargets, ndim] )
    if rank == 0:
        print ("Number of observation points: {0}".format(ntargets) )
    misfit = PointwiseStateObservation(Vh[STATE], targets)
    
    gamma = .1
    delta = .5
    
    theta0 = 2.
    theta1 = .5
    alpha  = math.pi/4
    
    anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree = 1)
    anis_diff.set(theta0, theta1, alpha)
    
    prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, anis_diff, robin_bc=True )
    
    mtrue = true_model(prior)
            
    if rank == 0:
        print ( "Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,2) )   
                
    #Generate synthetic observations
    utrue = pde.generate_state()
    x = [utrue, mtrue, None]
    pde.solveFwd(x[STATE], x)
    misfit.B.mult(x[STATE], misfit.d)
    rel_noise = 0.01
    MAX = misfit.d.norm("linf")
    noise_std_dev = rel_noise * MAX
    parRandom.normal_perturb(noise_std_dev, misfit.d)
    misfit.noise_variance = noise_std_dev*noise_std_dev
    
    model = Model(pde,prior, misfit)
    
    if rank == 0:
        print( sep, "Test the gradient and the Hessian of the model", sep )
    
    m0 = dl.interpolate(dl.Expression("sin(x[0])", element=Vh[PARAMETER].ufl_element() ), Vh[PARAMETER])
    modelVerify(model, m0.vector(), is_quadratic = False, verbose = (rank == 0) )

    if rank == 0:
        print( sep, "Find the MAP point", sep)
    m = prior.mean.copy()
    parameters = ReducedSpaceNewtonCG_ParameterList()
    parameters["rel_tolerance"] = 1e-9
    parameters["abs_tolerance"] = 1e-12
    parameters["max_iter"]      = 25
    parameters["globalization"] = "LS"
    parameters["GN_iter"] = 5
    if rank != 0:
        parameters["print_level"] = -1
        
    if rank == 0:
        parameters.showMe()
    solver = ReducedSpaceNewtonCG(model, parameters)
    
    x = solver.solve([None, m, None])
    
    if rank == 0:
        if solver.converged:
            print( "\nConverged in ", solver.it, " iterations.")
        else:
            print( "\nNot Converged")

        print ("Termination reason: ", solver.termination_reasons[solver.reason])
        print ("Final gradient norm: ", solver.final_grad_norm)
        print ("Final cost: ", solver.final_cost)
        
    if rank == 0:
        print (sep, "Compute the low rank Gaussian Approximation of the posterior", sep)
    
    model.setPointForHessianEvaluations(x, gauss_newton_approx = False)
    Hmisfit = ReducedHessian(model, misfit_only=True)
    
    return Hmisfit, prior.R, prior.Rsolver
        
def run(Hop, R, Rsolver, k_evec, p_evec):
    comm = R.mpi_comm()
    mpi_rank = dl.MPI.rank(comm)
    mpi_size = dl.MPI.size(comm)

    myRandom = Random(mpi_rank, mpi_size)

    x_vec = dl.Vector(comm)
    Hop.init_vector(x_vec,1)

    Omega_ref = MultiVector(x_vec,k_evec+p_evec)
    myRandom.normal(1., Omega_ref)
    k_evec = k_evec
    
    d_true,U_true = doublePassG(Hop,R,Rsolver,MultiVector(Omega_ref),k_evec,s=3)
    d_dp,U_dp = doublePassG(Hop,R,Rsolver,MultiVector(Omega_ref),k_evec,s=1)
    d_our, U_our  = singlePassG(Hop,R,Rsolver,MultiVector(Omega_ref),k_evec,s=1)
    d_their, U_their = singlePassG_original(Hop,R,Rsolver,MultiVector(Omega_ref),k_evec)
    
    d =  np.zeros((k_evec, 5), dtype=np.float64)
    
    d[:,0] =  np.arange(k_evec)+1
    d[:,1] =  d_true
    d[:,2] =  d_dp
    d[:,3] =  d_our
    d[:,4] =  d_their
    
    
    fname  = 'data_single_pass_comparisons_{0:d}_{1:d}.txt'.format(k_evec, p_evec)
    np.savetxt(fname, d, header='eigenval_index exact double_pass our_single_pass saibaba_single_pass')
    
    if mpi_rank==0:
        plt.figure()
        plt.semilogy(d_true, 's-k', label='Exact')
        plt.semilogy(d_dp, '*-b', label='Double Pass')
        plt.semilogy(d_their, '^-r', label='Single Pass (SaibabaEtAl)')
        plt.semilogy(d_our, 'v-g', label='Single Pass (Our)')
        plt.ylim([1e-2, 5e4])
        plt.legend()
        #plt.show()

    if mpi_rank==0:
        plt.figure()
        plt.semilogy(np.abs(d_true-d_dp), 's-k', label='Exact - DP')
        plt.semilogy(np.abs(d_true - d_their), '^-r', label='Single Pass (SaibabaEtAl)')
        plt.semilogy(np.abs(d_true - d_our), 'v-g', label='Single Pass (Our)')
        #plt.ylim([1e-2, 5e4])
        plt.legend()
        #plt.show()


if __name__ == "__main__":
    
    Hop, R, Rsolver = getHessian(64, 64)
    k_evec = 30
    
    run(Hop, R, Rsolver, k_evec, p_evec = 5)
    run(Hop, R, Rsolver, k_evec, p_evec = 10)
    run(Hop, R, Rsolver, k_evec, p_evec = 20)
    
        
    