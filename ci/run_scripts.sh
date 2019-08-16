#!/bin/bash

set -ev

ls
sed -i 's/nx = 64/nx = 32/' applications/poisson/model_subsurf.py
sed -i 's/ny = 64/ny = 32/' applications/poisson/model_subsurf.py
sed -i 's/nx = 64/nx = 32/' applications/poisson/model_continuous_obs.py
sed -i 's/ny = 64/ny = 32/' applications/poisson/model_continuous_obs.py
sed -i 's/mesh = dl.Mesh("ad_10k.xml")/mesh = dl.Mesh("ad_20.xml")/' applications/ad_diff/model_ad_diff.py
sed -i 's/nx = 64/nx = 32/' applications/mcmc/model_subsurf.py
sed -i 's/ny = 64/ny = 32/' applications/mcmc/model_subsurf.py
sed -i 's/chain.parameters["number_of_samples"] = 100/chain.parameters["number_of_samples"] = 30/' applications/mcmc/model_subsurf.py

echo $FENICS_VERSION

if [ "$FENICS_VERSION" == "2016.2" ]; then

    PYTHON=python
    QUAY=quay.io/fenicsproject/stable:2016.2.0 
    PYTHON_PREPROC=""

else

    PYTHON=python3
    QUAY=quay.io/fenicsproject/stable:2017.2.0 
    PYTHON_PREPROC="export MPLBACKEND=Agg; export hIPPYlibDeprecationWarning=error;"
fi

DOCKER="docker run --rm -v $(pwd):/home/fenics/hippylib -w"

${DOCKER} /home/fenics/hippylib $QUAY "dolfin-version"
${DOCKER} /home/fenics/hippylib/applications/poisson $QUAY "$PYTHON_PREPROC mpirun -n 2 $PYTHON model_continuous_obs.py"
${DOCKER} /home/fenics/hippylib/applications/poisson $QUAY "$PYTHON_PREPROC mpirun -n 2 $PYTHON model_subsurf.py"
${DOCKER} /home/fenics/hippylib/applications/ad_diff $QUAY "$PYTHON_PREPROC mpirun -n 1 $PYTHON model_ad_diff.py"
${DOCKER} /home/fenics/hippylib/applications/mcmc    $QUAY "$PYTHON_PREPROC mpirun -n 1 $PYTHON model_subsurf.py"

if [ "$FENICS_VERSION" == "2017.2" ]; then

    ${DOCKER} /home/fenics/hippylib/tutorial $QUAY "jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 1_FEniCS101.ipynb"
    ${DOCKER} /home/fenics/hippylib/tutorial $QUAY "jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 2_PoissonDeterministic.ipynb"
    ${DOCKER} /home/fenics/hippylib/tutorial $QUAY "jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 3_SubsurfaceBayesian.ipynb"
    ${DOCKER} /home/fenics/hippylib/tutorial $QUAY "jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 4_AdvectionDiffusionBayesian.ipynb"
    ${DOCKER} /home/fenics/hippylib/tutorial $QUAY "jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 5_HessianSpectrum.ipynb"

fi
