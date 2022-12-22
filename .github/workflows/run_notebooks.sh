PYTHON=python3


cd /__w/hippylib/hippylib/tutorial
jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 1_FEniCS101.ipynb
jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 2_PoissonDeterministic.ipynb
jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 3_SubsurfaceBayesian.ipynb
jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 4_AdvectionDiffusionBayesian.ipynb
jupyter nbconvert --ExecutePreprocessor.kernel_name="python3" --ExecutePreprocessor.timeout=1200 --to html --execute 5_HessianSpectrum.ipynb
