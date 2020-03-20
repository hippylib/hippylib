#!/bin/bash

set -ev 


PYTHON=python3

DOCKER="docker run --rm -v $(pwd):/home/fenics/hippylib -w"

${DOCKER} /home/fenics/hippylib/hippylib/test $IMAGE "$PYTHON -m unittest discover -v"
${DOCKER} /home/fenics/hippylib/hippylib/test $IMAGE "$PYTHON -m unittest discover -v -p 'ptest_*' "

${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c 'import hippylib' && cd hippylib/test/ && mpirun -n 4 $PYTHON ptest_pointwise_observation.py"
${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c 'import hippylib' && cd hippylib/test/ && mpirun -n 2 $PYTHON ptest_multivector.py"
${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c 'import hippylib' && cd hippylib/test/ && mpirun -n 2 $PYTHON ptest_randomizedSVD.py"
${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c 'import hippylib' && cd hippylib/test/ && mpirun -n 2 $PYTHON ptest_randomizedEigensolver.py"
${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c 'import hippylib' && cd hippylib/test/ && mpirun -n 2 $PYTHON ptest_qoi.py"
${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c 'import hippylib' && cd hippylib/test/ && mpirun -n 2 $PYTHON ptest_collectives.py "
${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c 'import hippylib' && cd hippylib/test/ && mpirun -n 2 $PYTHON ptest_numpy2expression.py"
${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c 'import hippylib' && cd hippylib/test/ && mpirun -n 2 $PYTHON ptest_numpy2meshFunction.py"

#${DOCKER} /home/fenics/hippylib/hippylib/test $IMAGE "mpirun -n 4 $PYTHON -m unittest discover -v -p 'ptest_*' "
