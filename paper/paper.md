---
title: 'hIPPYlib: An Extensible Software Framework for Large-Scale Inverse Problems'
tags:
  - Python
  - Infinite-dimensional inverse problems
  - adjoint-based methods
  - numerical optimization
  - Bayesian inference
  - uncertainty quantification
  - PDE toolkit
authors:
  - name: Umberto Villa
    affiliation: 1
  - name: Noemi Petra
    affiliation: 2
  - name: Omar Ghattas
    affiliation: 3
affiliations:
  - name: Institute for Computational Engineering & Sciences, The University of Texas at Austin
    index: 1
  - name: Applied Mathematics, School of Natural Sciences, University of California, Merced
    index: 2
  - name: Institute for Computational Engineering & Sciences, Department of Mechanical Engineering, and Department of Geological Sciences, The University of Texas at Austin
    index: 3
date: 1 June 2018
bibliography: paper.bib
---

# Summary

[`hIPPYlib`](https://hippylib.github.io) [@VillaPetraGhattas18] implements
state-of-the-art scalable algorithms for PDE-based deterministic and Bayesian inverse problems.
It builds on [`FEniCS`](http://fenicsproject.org/)
(a parallel finite element element library, [@LoggWells10,@LoggMardalGarth12,@LangtangenLogg17])
for the discretization of the PDE and on [`PETSc`](http://www.mcs.anl.gov/petsc/)
for scalable and efficient linear algebra operations and solvers.

Conceptually, `hIPPYlib` can be viewed as a toolbox that provides the
building blocks for experimenting new ideas and developing scalable
algorithms for PDE-based deterministic and Bayesian inverse problems.

In `hIPPYlib` the user can express the forward PDE and the likelihood in
weak form using the friendly, compact, near-mathematical notation of
`FEniCS`, which will then automatically generate efficient code for the
discretization.  Linear and nonlinear, and stationary and
time-dependent PDEs are supported in `hIPPYlib`. Currently, gradient and
Hessian information needs to be provided by the user; our future plan
includes automatic generation of this information by using `FEniCS`
automatic differentiation capabilities for users who do not wish to
derive the relevant weak forms.

Noise and prior covariance operators are modeled as inverses of
elliptic differential operators allowing us to build on existing fast
multigrid solvers for elliptic operators without explicitly
constructing the dense covariance operator.

`hIPPYlib` provides a robust implementation of the inexact
Newton-conjugate gradient algorithm to compute the maximum a posterior
(MAP) point. The reduced gradient and Hessian actions are
automatically computed via their weak form specification in `FEniCS` by
constraining the state and adjoint variables to satisfy the forward
and adjoint problem. The Newton system is solved inexactly by early
termination of CG iterations via Eisenstat-Walker (to prevent
oversolving) and Steihaug (to avoid negative curvature)
criteria. Globalization is achieved with an Armijo back-tracking line
search.

`hIPPYlib` offers different scalable methods to sample from the prior
distribution: Krylov methods to approximate the action of a matrix
square root on a vector, the conjugate gradient sampler, and finally
samplers that exploit the finite element assembly procedure of the
inverse covariance operator to obtain a symmetric decomposition. To
sample from a local Gaussian approximation (such as at the MAP point)
`hIPPYlib` exploits the low rank factorization of the Hessian misfit to
correct samples from the prior distribution.

Finally, randomized and probing algorithms are available to compute
the variance of the prior/posterior distribution.

The source code for `hIPPYlib` has been archived to Zenodo with the linked DOI: [@zenodo].


#Acknowledgements

`hIPPYlib` development is partially supported by the U.S. National Science Foundation (NSF), Software Infrastructure for Sustained Innovation (SI2: SSE & SSI) Program under grants ACI-1550593, ACI-1550547.

#References
