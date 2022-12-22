#!/bin/bash

set -ev 


PYTHON=python3

cd /__w/hippylib/hippylib/hippylib/test
$PYTHON -m unittest discover -v
$PYTHON -m unittest discover -v -p 'ptest_*'

mpirun -n 4 $PYTHON ptest_pointwise_observation.py
mpirun -n 2 $PYTHON ptest_multivector.py
mpirun -n 2 $PYTHON ptest_randomizedSVD.py
mpirun -n 2 $PYTHON ptest_randomizedEigensolver.py
mpirun -n 2 $PYTHON ptest_qoi.py
mpirun -n 2 $PYTHON ptest_collectives.py
mpirun -n 2 $PYTHON ptest_numpy2expression.py
mpirun -n 2 $PYTHON ptest_numpy2meshFunction.py
