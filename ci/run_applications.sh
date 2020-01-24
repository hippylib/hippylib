#!/bin/bash

set -ev

ls
PYTHON=python3
PYTHON_PREPROC="export MPLBACKEND=agg; export hIPPYlibDeprecationWarning=error; cd /home/fenics/hippylib; $PYTHON -c 'import hippylib'; cd -;"



DOCKER="docker run --rm -v $(pwd):/home/fenics/hippylib -w"

${DOCKER} /home/fenics/hippylib $IMAGE "dolfin-version"
${DOCKER} /home/fenics/hippylib/applications/poisson    $IMAGE "$PYTHON_PREPROC mpirun -n 2 $PYTHON model_continuous_obs.py --nx 32 --ny 32"
${DOCKER} /home/fenics/hippylib/applications/poisson    $IMAGE "$PYTHON_PREPROC mpirun -n 2 $PYTHON model_subsurf.py --nx 32 --ny 32 --nsamples 5"
${DOCKER} /home/fenics/hippylib/applications/ad_diff    $IMAGE "$PYTHON_PREPROC mpirun -n 2 $PYTHON model_ad_diff.py --mesh ad_20.xml"
${DOCKER} /home/fenics/hippylib/applications/mcmc       $IMAGE "$PYTHON_PREPROC mpirun -n 2 $PYTHON model_subsurf.py --nx 32 --ny 32 --nsamples 30"
${DOCKER} /home/fenics/hippylib/applications/forward_uq $IMAGE "$PYTHON_PREPROC mpirun -n 2 $PYTHON model_subsurf_effperm.py --nx 16 --ny 32 --nsamples 30"
