# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append( "../../" )
from hippylib import *


class TimeDependentAD:    
    def __init__(self, mesh, Vh, t_init, t_final, t_1, dt, wind_velocity, gls_stab, Prior):
        self.mesh = mesh
        self.Vh = Vh
        self.t_init = t_init
        self.t_final = t_final
        self.t_1 = t_1
        self.dt = dt
        self.sim_times = np.arange(self.t_init, self.t_final+.5*self.dt, self.dt)
        
        u = dl.TrialFunction(Vh[STATE])
        v = dl.TestFunction(Vh[STATE])
        
        kappa = dl.Constant(.001)
        dt_expr = dl.Constant(self.dt)
        
        r_trial = u + dt_expr*( -dl.div(kappa*dl.nabla_grad(u))+ dl.inner(wind_velocity, dl.nabla_grad(u)) )
        r_test  = v + dt_expr*( -dl.div(kappa*dl.nabla_grad(v))+ dl.inner(wind_velocity, dl.nabla_grad(v)) )

        
        h = dl.CellSize(mesh)
        vnorm = dl.sqrt(dl.inner(wind_velocity, wind_velocity))
        if gls_stab:
            tau = dl.Min((h*h)/(dl.Constant(2.)*kappa), h/vnorm )
        else:
            tau = dl.Constant(0.)
                            
        self.M = dl.assemble( dl.inner(u,v)*dl.dx )
        self.M_stab = dl.assemble( dl.inner(u, v+tau*r_test)*dl.dx )
        self.Mt_stab = dl.assemble( dl.inner(u+tau*r_trial,v)*dl.dx )
        Nvarf  = (dl.inner(kappa *dl.nabla_grad(u), dl.nabla_grad(v)) + dl.inner(wind_velocity, dl.nabla_grad(u))*v )*dl.dx
        Ntvarf  = (dl.inner(kappa *dl.nabla_grad(v), dl.nabla_grad(u)) + dl.inner(wind_velocity, dl.nabla_grad(v))*u )*dl.dx
        self.N  = dl.assemble( Nvarf )
        self.Nt = dl.assemble(Ntvarf)
        stab = dl.assemble( tau*dl.inner(r_trial, r_test)*dl.dx)
        self.L = self.M + dt*self.N + stab
        self.Lt = self.M + dt*self.Nt + stab
        
        boundaries = dl.FacetFunction("size_t", mesh)
        boundaries.set_all(0)

        class InsideBoundary(dl.SubDomain):
            def inside(self,x,on_boundary):
                x_in = x[0] > dl.DOLFIN_EPS and x[0] < 1 - dl.DOLFIN_EPS
                y_in = x[1] > dl.DOLFIN_EPS and x[1] < 1 - dl.DOLFIN_EPS
                return on_boundary and x_in and y_in
            
        Gamma_M = InsideBoundary()
        Gamma_M.mark(boundaries,1)
        ds_marked = dl.Measure("ds")[boundaries]
        
        self.Q = dl.assemble( self.dt*dl.inner(u, v) * ds_marked(1) )

        self.Prior = Prior
        
        self.solver = dl.PETScKrylovSolver("gmres", "ilu")
        self.solver.set_operator(self.L)
        
        self.solvert = dl.PETScKrylovSolver("gmres", "ilu")
        self.solvert.set_operator(self.Lt)
                        
        self.ud = self.generate_vector(STATE)
        self.noise_variance = 0
                
    def generate_vector(self, component = "ALL"):
        if component == "ALL":
            u = TimeDependentVector(self.sim_times)
            u.initialize(self.Q, 0)
            a = dl.Vector()
            self.Prior.init_vector(a,0)
            p = TimeDependentVector(self.sim_times)
            p.initialize(self.Q, 0)
            return [u, a, p]
        elif component == STATE:
            u = TimeDependentVector(self.sim_times)
            u.initialize(self.Q, 0)
            return u
        elif component == PARAMETER:
            a = dl.Vector()
            self.Prior.init_vector(a,0)
            return a
        elif component == ADJOINT:
            p = TimeDependentVector(self.sim_times)
            p.initialize(self.Q, 0)
            return p
        else:
            raise
    
    def init_parameter(self, a):
        self.Prior.init_vector(a,0)
        
    def getIdentityMatrix(self, component):
        Xh = self.Vh[component]
        test = dl.TestFunction(Xh)
        trial = dl.TrialFunction(Xh)
        
        I = dl.assemble(test*trial*dl.dx)
        I.zero()
        I.ident_zeros()
        
        return I
        
          
    def cost(self, x):
        Rdx = dl.Vector()
        self.Prior.init_vector(Rdx,0)
        dx = x[PARAMETER] - self.Prior.mean
        self.Prior.R.mult(dx, Rdx)
        reg = .5*Rdx.inner(dx)
        
        u  = dl.Vector()
        ud = dl.Vector()
        self.Q.init_vector(u,0)
        self.Q.init_vector(ud,0)
    
        misfit = 0
        for t in np.arange(self.t_1, self.t_final+(.5*self.dt), self.dt):
            x[STATE].retrieve(u,t)
            self.ud.retrieve(ud,t)
            diff = u - ud
            Qdiff = self.Q * diff
            misfit += .5/self.noise_variance*Qdiff.inner(diff)
            
        c = misfit + reg
                
        return [c, reg, misfit]
    
    def solveFwd(self, out, x, tol=1e-9):
        out.zero()
        uold = x[PARAMETER]
        u = dl.Vector()
        rhs = dl.Vector()
        self.M.init_vector(rhs, 0)
        self.M.init_vector(u, 0)
        self.solver.parameters["relative_tolerance"] = tol
        t = self.t_init
        while t < self.t_final:
            t += self.dt
            self.M_stab.mult(uold, rhs)
            self.solver.solve(u, rhs)
            out.store(u,t)
            uold = u
    
    def solveAdj(self, out, x, tol=1e-9):
        out.zero()
        pold = dl.Vector()
        self.M.init_vector(pold,0)    
        p = dl.Vector()
        self.M.init_vector(p,0)
        rhs = dl.Vector()
        self.M.init_vector(rhs,0) 
        rhs_obs = dl.Vector()
        
        u = dl.Vector()
        self.M.init_vector(u,0)
        ud = dl.Vector()
        self.M.init_vector(ud,0)
  
        self.solvert.parameters["relative_tolerance"] = tol
                
        t = self.t_final
        while t > self.t_init:
            self.Mt_stab.mult(pold,rhs)
            if t > self.t_1 - .5*self.dt:
                x[STATE].retrieve(u,t)
                self.ud.retrieve(ud,t)
                ud.axpy(-1., u)
                self.Q.mult(ud,rhs_obs)
#                print "t = ", t, "solveAdj ||ud-u||_inf = ", ud.norm("linf"), " ||rhs_obs|| = ", rhs_obs.norm("linf")
                rhs.axpy(1./self.noise_variance, rhs_obs)
                
            self.solvert.solve(p, rhs)
            pold = p
            out.store(p, t)
            t -= self.dt
            
            
            
    def evalGradientParameter(self,x, mg):
        self.Prior.init_vector(mg,1)
        dx = x[PARAMETER] - self.Prior.mean
        self.Prior.R.mult(dx, mg)
        
        p0 = dl.Vector()
        self.Q.init_vector(p0,0)
        x[ADJOINT].retrieve(p0, self.t_init + self.dt)
        
        mg.axpy(-1., self.Mt_stab*p0)
        
        g = dl.Vector()
        self.M.init_vector(g,1)
        
        self.Prior.Msolver.solve(g,mg)
        
        
        grad_norm = g.inner(mg)
        
        return grad_norm
        
    
    def setPointForHessianEvaluations(self, x):
        """
        Specify the point x = [u,a,p] at which the Hessian operator (or the Gauss-Newton approximation)
        need to be evaluated.
        
        Nothing to do since the problem is linear
        """
        return

        
    def solveFwdIncremental(self, sol, rhs, tol):
        sol.zero()
        uold = dl.Vector()
        u = dl.Vector()
        Muold = dl.Vector()
        myrhs = dl.Vector()
        self.M.init_vector(uold, 0)
        self.M.init_vector(u, 0)
        self.M.init_vector(Muold, 0)
        self.M.init_vector(myrhs, 0)
        self.solver.parameters["relative_tolerance"] = tol
        t = self.t_init
        while t < self.t_final:
            t += self.dt
            self.M_stab.mult(uold, Muold)
            rhs.retrieve(myrhs, t)
            myrhs.axpy(1., Muold)
            self.solver.solve(u, myrhs)
            sol.store(u,t)
            uold = u


        
    def solveAdjIncremental(self, sol, rhs, tol):
        sol.zero()
        pold = dl.Vector()
        p = dl.Vector()
        Mpold = dl.Vector()
        myrhs = dl.Vector()
        self.M.init_vector(pold, 0)
        self.M.init_vector(p, 0)
        self.M.init_vector(Mpold, 0)
        self.M.init_vector(myrhs, 0)
        self.solvert.parameters["relative_tolerance"] = tol
        t = self.t_final
        while t > self.t_init:
            self.Mt_stab.mult(pold, Mpold)
            rhs.retrieve(myrhs, t)
            myrhs.axpy(1., Mpold)
            self.solvert.solve(p, myrhs)
            sol.store(p,t)
            pold = p
            t -= self.dt
    
    def applyC(self, da, out):
        out.zero()
        myout = dl.Vector()
        self.M.init_vector(myout, 0)
        self.M_stab.mult(da,myout)
        myout *= -1.
        t = self.t_init + self.dt
        out.store(myout,t)
        
        myout.zero()
        while t < self.t_final:
            t += self.dt
            out.store(myout,t)
    
    def applyCt(self, dp, out):
        t = self.t_init + self.dt
        dp0 = dl.Vector()
        self.M.init_vector(dp0,0)
        dp.retrieve(dp0, t)
        dp0 *= -1.
        self.Mt_stab.mult(dp0, out)

    
    def applyWuu(self, du, out, gn_approx=False):
        out.zero()
        myout = dl.Vector()
        self.Q.init_vector(myout,0)
        myout.zero()
        
        t = self.t_init + self.dt
        while t < self.t_1 - .5*self.dt:
            out.store(myout, t)
            t += self.dt
            
        mydu  = dl.Vector()
        self.Q.init_vector(mydu,0)
        while t < self.t_final+(.5*self.dt):
            du.retrieve(mydu,t)
            self.Q.mult(mydu, myout)
            myout *= 1./self.noise_variance
            out.store(myout, t)
            t += self.dt
    
    def applyWua(self, da, out):
        out.zero()

    
    def applyWau(self, du, out):
        out.zero()
    
    def applyR(self, da, out):
        self.Prior.R.mult(da,out)
    
    def applyRaa(self, da, out):
        out.zero()
        
    def exportState(self, x, filename, varname):
        out_file = dl.File(filename)
        ufunc = dl.Function(self.Vh[STATE], name=varname)
        t = self.t_init
        out_file << (vector2Function(x[PARAMETER], self.Vh[STATE], name=varname),t)
        while t < self.t_final:
            t += self.dt
            x[STATE].retrieve(ufunc.vector(), t)
            out_file << (ufunc, t)
            
        
        
def v_boundary(x,on_boundary):
    return on_boundary

def q_boundary(x,on_boundary):
    return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS
        
def computeVelocityField(mesh):
    Xh = dl.VectorFunctionSpace(mesh,'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    XW = dl.MixedFunctionSpace([Xh, Wh])

    
    Re = 1e2
    
    g = dl.Expression(('0.0','(x[0] < 1e-14) - (x[0] > 1 - 1e-14)'))
    bc1 = dl.DirichletBC(XW.sub(0), g, v_boundary)
    bc2 = dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise')
    bcs = [bc1, bc2]
    
    vq = dl.Function(XW)
    (v,q) = dl.split(vq)
    (v_test, q_test) = dl.TestFunctions (XW)
    
    def strain(v):
        return dl.sym(dl.nabla_grad(v))
    
    F = ( (2./Re)*dl.inner(strain(v),strain(v_test))+ dl.inner (dl.nabla_grad(v)*v, v_test)
           - (q * dl.div(v_test)) + ( dl.div(v) * q_test) ) * dl.dx
           
    dl.solve(F == 0, vq, bcs, solver_parameters={"newton_solver":
                                         {"relative_tolerance":1e-4, "maximum_iterations":100}})
        
    return v
    

        
if __name__ == "__main__":
    dl.set_log_active(False)
    np.random.seed(1)
    sep = "\n"+"#"*80+"\n"
    print sep, "Set up the mesh and finite element spaces.\n","Compute wind velocity", sep
    mesh = dl.refine( dl.Mesh("ad_20.xml") )
    wind_velocity = computeVelocityField(mesh)
    Vh = dl.FunctionSpace(mesh, "Lagrange", 2)
    print "Number of dofs: {0}".format( Vh.dim() )
    
    print sep, "Set up Prior Information and model", sep
    
    true_initial_condition = dl.interpolate(dl.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))'), Vh).vector()

    orderPrior = 2
    
    if orderPrior == 1:
        gamma = 1
        delta = 1e1
        prior = LaplacianPrior(Vh, gamma, delta)
    elif orderPrior == 2:
        gamma = 1
        delta = 8
        prior = BiLaplacianPrior(Vh, gamma, delta)
        
#    prior.mean = interpolate(Expression('min(0.6,exp(-50*(pow(x[0]-0.34,2) +  pow(x[1]-0.71,2))))'), Vh).vector()
    prior.mean = dl.interpolate(dl.Expression('0.5'), Vh).vector()
    
    print "Prior regularization: (delta - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,orderPrior)

    problem = TimeDependentAD(mesh, [Vh,Vh,Vh], 0., 4., 1., .2, wind_velocity, True, prior)
    
    print sep, "Generate synthetic observation", sep
    rel_noise = 0.001
    utrue = problem.generate_vector(STATE)
    x = [utrue, true_initial_condition, None]
    problem.solveFwd(x[STATE], x, 1e-9)
    MAX = utrue.norm("linf", "linf")
    noise_std_dev = rel_noise * MAX
    problem.ud.copy(utrue)
    problem.ud.randn_perturb(noise_std_dev)
    problem.noise_variance = noise_std_dev*noise_std_dev
    
    print sep, "Test the gradient and the Hessian of the model", sep
    a0 = true_initial_condition.copy()
    modelVerify(problem, a0, 1e-12, is_quadratic = True)
    
    print sep, "Compute the reduced gradient and hessian", sep
    [u,a,p] = problem.generate_vector()
    problem.solveFwd(u, [u,a,p], 1e-12)
    problem.solveAdj(p, [u,a,p], 1e-12)
    mg = problem.generate_vector(PARAMETER)
    grad_norm = problem.evalGradientParameter([u,a,p], mg)
        
    print "(g,g) = ", grad_norm
    
    H = ReducedHessian(problem, 1e-12, gauss_newton_approx=False, misfit_only=True) 
    
    print sep, "Compute the low rank Gaussian Approximation of the posterior", sep   
    k = 80
    p = 20
    print "Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p)
    Omega = np.random.randn(a.array().shape[0], k+p)
    d, U = singlePassG(H, prior.R, prior.Rsolver, Omega, k, check_Bortho=False, check_Aortho=False, check_residual=False)
    posterior = GaussianLRPosterior( prior, d, U )
    
    print sep, "Find the MAP point", sep
    
    H.misfit_only = False
        
    solver = CGSolverSteihaug()
    solver.set_operator(H)
    solver.set_preconditioner( posterior.Hlr )
    solver.parameters["print_level"] = 1
    solver.parameters["rel_tolerance"] = 1e-6
    solver.solve(a, -mg)
    problem.solveFwd(u, [u,a,p], 1e-12)
 
    total_cost, reg_cost, misfit_cost = problem.cost([u,a,p])
    print "Total cost {0:5g}; Reg Cost {1:5g}; Misfit {2:5g}".format(total_cost, reg_cost, misfit_cost)
    
    posterior.mean = a

    compute_trace = False
    if compute_trace:
        post_tr, prior_tr, corr_tr = posterior.trace(method="Exact", tol=5e-2, min_iter=20, max_iter=100)
        print "Posterior trace {0:5g}; Prior trace {1:5g}; Correction trace {2:5g}".format(post_tr, prior_tr, corr_tr)
        post_pw_variance, pr_pw_variance, corr_pw_variance = posterior.pointwise_variance("Exact")
    
    print sep, "Save results", sep  
    problem.exportState([u,a,p], "results/conc.pvd", "concentration")
    problem.exportState([utrue,true_initial_condition,p], "results/true_conc.pvd", "concentration")
    problem.exportState([problem.ud,true_initial_condition,p], "results/noisy_conc.pvd", "concentration")

    if compute_trace:
        fid = dl.File("results/pointwise_variance.pvd")
        fid << vector2Function(post_pw_variance, Vh, name="Posterior")
        fid << vector2Function(pr_pw_variance, Vh, name="Prior")
        fid << vector2Function(corr_pw_variance, Vh, name="Correction")
    
    posterior.exportU(Vh, "hmisfit/evect.pvd")
    np.savetxt("hmisfit/eigevalues.dat", d)
    
    
    
    print sep, "Generate samples from Prior and Posterior", sep
    fid_prior = dl.File("samples/sample_prior.pvd")
    fid_post  = dl.File("samples/sample_post.pvd")
    nsamples = 500
    noise = dl.Vector()
    posterior.init_vector(noise,"noise")
    noise_size = noise.array().shape[0]
    s_prior = dl.Function(Vh, name="sample_prior")
    s_post = dl.Function(Vh, name="sample_post")
    for i in range(nsamples):
        noise.set_local( np.random.randn( noise_size ) )
        posterior.sample(noise, s_prior.vector(), s_post.vector())
        fid_prior << s_prior
        fid_post << s_post
    
    
    print sep, "Visualize results", sep 
    plt.figure()
    plt.plot(range(0,k), d, 'b*', range(0,k), np.ones(k), '-r')
    plt.yscale('log')
    dl.plot(vector2Function(a, Vh, name = "Initial Condition"))
    plt.show()
    dl.interactive()

    
