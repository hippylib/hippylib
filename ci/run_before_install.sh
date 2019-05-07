#!/bin/bash

set -ev

echo $FENICS_VERSION

if [ "$FENICS_VERSION" == "2018.1" ]; then

    docker pull quay.io/fenicsproject/stable:2018.1.0

elif [ "$FENICS_VERSION" == "2019.1" ]; then

    docker pull quay.io/fenicsproject/stable:2019.1.0

fi

