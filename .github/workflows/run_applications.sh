#!/bin/bash

set -ev

PYTHON=python3
export MPLBACKEND=agg
export hIPPYlibDeprecationWarning=error
$PYTHON -c 'import hippylib'

cd applications/poisson  && mpirun -n 2 $PYTHON model_continuous_obs.py --nx 32 --ny 32
cd ../poisson  && mpirun -n 2 $PYTHON model_subsurf.py --nx 32 --ny 32 --nsamples 5
cd ../ad_diff  && mpirun -n 2 $PYTHON model_ad_diff.py --mesh ad_20.xml
cd ../mcmc     && mpirun -n 2 $PYTHON model_subsurf.py --nx 32 --ny 32 --nsamples 30
cd ../forward_uq && mpirun -n 2 $PYTHON model_subsurf_effperm.py --nx 16 --ny 32 --nsamples 30
