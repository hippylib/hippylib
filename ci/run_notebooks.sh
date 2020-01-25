PYTHON=python3

DOCKER="docker run --rm -v $(pwd):/home/fenics/hippylib -w"

${DOCKER} /home/fenics/hippylib/tutorial $IMAGE "jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 1_FEniCS101.ipynb"
${DOCKER} /home/fenics/hippylib/tutorial $IMAGE "jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 2_PoissonDeterministic.ipynb"
${DOCKER} /home/fenics/hippylib/tutorial $IMAGE "jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 3_SubsurfaceBayesian.ipynb"
${DOCKER} /home/fenics/hippylib/tutorial $IMAGE "jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 4_AdvectionDiffusionBayesian.ipynb"
${DOCKER} /home/fenics/hippylib/tutorial $IMAGE "jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 5_HessianSpectrum.ipynb"
