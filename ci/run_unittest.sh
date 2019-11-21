#!/bin/bash

set -ev 


PYTHON=python3

IMAGE=hippylib/fenics:latest

DOCKER="docker run --rm -v $(pwd):/home/fenics/hippylib -w"

${DOCKER} /home/fenics/hippylib/hippylib/test $IMAGE "$PYTHON -m unittest discover -v"
${DOCKER} /home/fenics/hippylib/hippylib/test $IMAGE "$PYTHON -m unittest discover -v -p 'ptest_*' "

${DOCKER} /home/fenics/hippylib/hippylib/test $IMAGE "mpirun -n 4 $PYTHON -m unittest discover -v -p 'ptest_*' "
