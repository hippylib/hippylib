#!/bin/bash

set -ev 

echo $FENICS_VERSION

if [ "$FENICS_VERSION" == "2018.1" ]; then

    PYTHON=python3
    QUAY=quay.io/fenicsproject/stable:2018.1.0 

elif [ "$FENICS_VERSION" == "2019.1" ]; then

    PYTHON=python3
    QUAY=quay.io/fenicsproject/stable:current 

fi

DOCKER="docker run --rm -v $(pwd):/home/fenics/hippylib -w"

${DOCKER} /home/fenics/hippylib/hippylib/test $QUAY "$PYTHON -m unittest discover -v"


