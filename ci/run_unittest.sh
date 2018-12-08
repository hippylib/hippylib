#!/bin/bash

set -ev 

echo $FENICS_VERSION

if [ "$FENICS_VERSION" == "2016.2" ]; then

    PYTHON=python
    QUAY=quay.io/fenicsproject/stable:2016.2.0 

elif [ "$FENICS_VERSION" == "2017.2" ]; then

    PYTHON=python3
    QUAY=quay.io/fenicsproject/stable:2017.2.0 

fi

DOCKER="docker run --rm -v $(pwd):/home/fenics/hippylib -w"

${DOCKER} /home/fenics/hippylib/hippylib/test $QUAY "$PYTHON test_gaussian_real_prior.py -v"
${DOCKER} /home/fenics/hippylib/hippylib/test $QUAY "$PYTHON test_deprecation_warnings.py -v"


