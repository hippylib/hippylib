#!/bin/bash

set -ev 


PYTHON=python3

IMAGE=hippylib/fenics:latest

DOCKER="docker run --rm -v $(pwd):/home/fenics/hippylib -w"

${DOCKER} /home/fenics/hippylib/hippylib/test $IMAGE "$PYTHON -m unittest discover -v"
${DOCKER} /home/fenics/hippylib/hippylib/test $IMAGE "$PYTHON -m unittest discover -v -p 'ptest_*' "

${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c 'import hippylib' && mpirun -n 4 $PYTHON hippylib/test/ptest_pointwise_observation.py"
${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c 'import hippylib' && mpirun -n 2 $PYTHON hippylib/test/ptest_multivector.py"
${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c 'import hippylib' && mpirun -n 2 $PYTHON hippylib/test/ptest_randomizedSVD.py"
${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c 'import hippylib' && mpirun -n 2 $PYTHON hippylib/test/ptest_randomizedEigensolver.py"

#${DOCKER} /home/fenics/hippylib/hippylib/test $IMAGE "mpirun -n 4 $PYTHON -m unittest discover -v -p 'ptest_*' "
