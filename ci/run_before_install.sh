#!/bin/bash

set -ev

docker pull hippylib/fenics:latest
docker pull quay.io/fenicsproject/stable:2019.1.0.r3
docker pull quay.io/fenicsproject/stable:master
