#!/bin/bash

set -ev

echo $FENICS_VERSION

if [ "$FENICS_VERSION" == "2016.2" ]; then

    docker pull quay.io/fenicsproject/stable:2016.2.0

elif [ "$FENICS_VERSION" == "2017.2" ]; then

    docker pull quay.io/fenicsproject/stable:2017.2.0

fi

