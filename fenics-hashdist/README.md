# Build FEniCS using Hashdist

This folder contains scripts and system specific profile files (MacOS
and Linux) to build FEniCS using Hashdist.  This script will build
FEniCS 2016.2 with all the suggested dependencies to fully exploit
hIPPYlib capabilities.

## Step-by-step build

1. *Select the correct hashdist profile file*

If you are running MacOS
```
ln -s local-darwin.yaml local.yaml
```
*Note*: You may have to install the Fortran compiler from [here](https://gcc.gnu.org/wiki/GFortranBinaries#MacOS).

If you are running Linux:
```
ln -s local-linux.yaml local.yaml
```

2. *Build FEniCS and all its dependencies*
```
chmod +x fenics-install.sh
./fenics-install.sh
```
This can take several hours.  When it completes, a file
`fenics.custom` will be generated.  This files contains all the paths
you need to add to your enviroment to run FEniCS.

3. *Source the fenics configuration file*

Everytime you open a new shell, you will have to add all the FEniCS
paths to your enviroment before you can use FEniCS.
```
source <HIPPYLIB_BASE_DIR>/fenics-hashdist/fenics.custom
```
where `<HIPPYLIB_BASE_DIR>` is the absolute path to the folder where
hIPPYlib resides.