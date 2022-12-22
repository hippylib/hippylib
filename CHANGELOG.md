                        Inverse Problem PYthon library

```
 __        ______  _______   _______   __      __  __  __  __       
/  |      /      |/       \ /       \ /  \    /  |/  |/  |/  |      
$$ |____  $$$$$$/ $$$$$$$  |$$$$$$$  |$$  \  /$$/ $$ |$$/ $$ |____  
$$      \   $$ |  $$ |__$$ |$$ |__$$ | $$  \/$$/  $$ |/  |$$      \ 
$$$$$$$  |  $$ |  $$    $$/ $$    $$/   $$  $$/   $$ |$$ |$$$$$$$  |
$$ |  $$ |  $$ |  $$$$$$$/  $$$$$$$/     $$$$/    $$ |$$ |$$ |  $$ |
$$ |  $$ | _$$ |_ $$ |      $$ |          $$ |    $$ |$$ |$$ |__$$ |
$$ |  $$ |/ $$   |$$ |      $$ |          $$ |    $$ |$$ |$$    $$/ 
$$/   $$/ $$$$$$/ $$/       $$/           $$/     $$/ $$/ $$$$$$$/  
```                                                                    
                                                                    

                          https://hippylib.github.io

Version 3.1.0, released on Dec 21, 2022
-------------------
- Switch to GitHub Actions for CI
- Introduce a new function `BiLaplacianComputeCoefficients` to estimate the PDE coefficient based on a
  prescribed marginal variance and correlation length
- Allow `BiLaplacianPrior` to take spatially varying coefficients as input
- Add support for non-Gaussian noise models
- Introduce utilities to interpolate cartesian data (expressed as `numpy arrays`) on a `dolfin` mesh. 

Version 3.0.0, released on Feb 2, 2020
---------------------------------------- 
- Support for `FEniCS 2019.1.0`
- Modify `PointwiseObservation` so that ordering of observations targets is respected also in parallel.
  Setting the flag `prune_and_sort` to `True` restores previous behavior.
- Remove unused input `tol` from `Model.solveFwd`, `Model.solveAdj`, `Model.solveFwdIncremental`, `Model.solveAdjIncremental`
  and from related classes.
- Use `argparse` to set parameters in application drivers from command line
- Use `dl.XDMFFile` to export solutions for visualization in Paraview in all application drivers
- Implement accuracy enhanced SVD algorithm in `randomizedSVD.py`
- Add forward UQ capabilities, using Taylor approximations as control variates
- Introduce hIPPYlib's wrappers to `petcs4py.PETSc.KSP`
- Add reduction operations useful when solving different PDEs concurrently on each process
- Increase coverage of unit testing in CI


Version 2.3.0, released on Sept 6, 2019
----------------------------------------
- Reimplement `BiLaplacianPrior` and `MollifiedBiLaplacianPrior` using a more general framework `SqrtPrecisionPDE_Prior`,
  which also supports Gaussian vector fields.
- Update the prior distribution in `model_subsurf.py` and `tutorial/3_SubsurfaceBayesian.ipynb` to use Robin boundary conditions to alleviate boundary artifacts.
- Update the data misfit term in `model_ad_diff.py` and `tutorial/4_AdvectionDiffusionBayesian.ipynb` to use discrete observations in space and time.
                          
Version 2.2.1, released on March 28, 2019
----------------------------------------                      
- Bug fix missing `mpi_comm` in `TimeDependentVector`
- Bug fix in the initialization of the global variable `parRandom`

Version 2.2.0, released on Dec 12, 2018
----------------------------------------
- Add new class `GaussianRealPrior` that implements a finite-dimensional Gaussian prior
- Add a `callback` user-defined function that can be called at the end of each inexact Newton CG iteration
- Add a `__version__` and `version_info` attribute to `hIPPYlib`
- Add `setup.py` to (optionally) install `hIPPYlib` via `pip`
- Add deprecation mechanism
- Deprecate `TimeDependentVector.copy(other)` in favor of `TimeDependentVector.copy()` for consistency with  `dolfin.Vector.copy`
- Deprecate `_BilaplacianR.inner(x,y)` for consistency with `dolfin.Matrix`
- CI enhancement via build matrix

Version 2.1.1, released on Oct 23, 2018
----------------------------------------
- Update `README.md` and `paper` according to JOSS reviewers's comments
- Add contributing guidelines
- Fix some typos in notebooks (thanks to **Christian Boehm**)

Version 2.1.0, released on July 18, 2018
----------------------------------------
- Alleviate boundary artifacts (inflation of marginal variance) in Bilaplacian-like priors
using Robin boundary conditions
- Allow the user to select different matplotlib colormaps in jupyter notebooks
- Buxfix in the acceptance ratio of the gpCN MCMC proposal
                          
Version 2.0.0, released on June 15, 2018
----------------------------------------
- Introduce capabilities for non-Gaussian Bayesian inference using Mark Chain Monte Carlo methods.
Kernels: mMALA, pCN, gpCN, IS. **Note: API subject to change**
- Support domain-decomposition parallelization (new parallel random number generator, and new randomized eigensolvers)
- The parameter, usually labeled `a`, throughout the library, has been renamed to `m`, for `model parameter`.
  Interface changes:
    - `PDEProblem.eval_da` --> `PDEProblem.evalGradientParameter`
    - `Model.applyWua` --> `Model.applyWum`
    - `Model.applyWau` --> `Model.applyWmu`
    - `Model.applyRaa` --> `Model.applyWmm`
    - `gda_tolerance` --> `gdm_tolerance` in the parameter list for Newton and QuasiNewton optimizers
    - `gn_approx` --> `gass_newton_approx` as parameter in function to compute Hessian/linearization point in classes `Model`, `PDEProblem`, `Misfit`, `Qoi`, `ReducedQoi`
- Organize `hippylib` in subpackages
- Add `sphinx` documentation (thanks to **E. Khattatov** and **I. Ambartsumyan**)

Version 1.6.0, released on May 16, 2018
----------------------------------------
- **Bugfix** in `PDEVariationalProblem.solveIncremental` for non self-adjoint models 
- Add new estimator for the trace and diagonal of the prior covariance
using randomized eigendecomposition
- In all examples and tutorial, use enviromental variable `HIPPYLIB_BASE_DIR` (if defined)
to add `hIPPYlib` to `PYTHONPATH`
                          
Version 1.5.0, released on Jan 24, 2018
----------------------------------------
- Add support for `FEniCS 2017.2`

Version 1.4.0, released on Nov 8, 2017
----------------------------------------
- Add support for `Python 3`
- Enchantments in `PDEVariationalProblem`: it now supports multiple Dirichlet
  condition and vectorial/mixed function spaces
- Bugfix: Set the correct number of global rows, when targets points fall 
  outside the computational domain
- More extensive testing with `Travis` Integration

Version 1.3.0, released on June 28, 2017
----------------------------------------
- Improve `hashdist` installation support
- Switch license to GPL-2
- Add support for `FEniCS 2017.1`

Version 1.2.0, released on April 24, 2017
----------------------------------------
- Update instruction to build `FEniCS`: `hashdist` and `docker`
- Update notebook to nbformat 4
- Let `FEniCS 2016.2` be the preferred version of `FEniCS`
- Add `Travis` integration
                          
Version 1.1.0, released on Nov 28, 2016
----------------------------------------

- Add partial support for `FEniCS 2016.1` (Applications and Tutorial)
- Improve performance of the randomized eigensolvers

Version 1.0.2, released on Sep 30, 2016
----------------------------------------

- Use `vector2Function` to safely convert `dolfin.Vector` to `dolfin.Function`
- Optimize the `PDEVariationalProblem` to exploit the case when the forward problem is linear
- Update notebook `1_FEniCS101.ipynb`
                           
Version 1.0.1, released on Aug 25, 2016
----------------------------------------

- Add support in `hippylib.Model` and `hippylib.Misfit` for misfit functional with explicit dependence on the parameter


Version 1.0.0, released on Aug 8, 2016
----------------------------------------

- Uploaded to https://hippylib.github.io.
- Initial release under GPL-3.
