#!/usr/bin/env sh
#
# This script installs FEniCS via HashDist.
# See README.rst for details.

# Tell script to exit on first error
set -e

# Check operating system
OS=$(uname)

# Check for Xcode and Xcode Command Line Tools on OS X
if [ "x$OS" = "xDarwin" ]; then
    if [ ! -x /usr/bin/xcodebuild ]; then
	echo >&2 "This script requires Xcode to be installed in order to run."
	echo >&2 "Please install Xcode from the Mac App Store and try again."
	exit 1
    fi

    set +e
    # 'xcode-select -p' return 2 if Xcode Command Line Tools are not installed
    xcode-select -p > /dev/null 2>&1
    if [ x$? = x2 ]; then
	echo >&2 "This script requires Xcode Command Line Tools to be installed in order to run."
	echo >&2 "Please install the Xcode Command Line Tools by running 'xcode-select --install' and try again."
	exit 1
    fi
    set -e
fi

# Check for git
hash git 2>/dev/null || \
{
    echo >&2 "This script requires git to run."
    echo >&2 "Try 'apt-get install git' or 'brew install git'."
    exit 1
}

# Use Python to get full path to a file (realpath is not available on OS X)
realpath() {
    echo $(python -c "import os,sys;print(os.path.realpath(sys.argv[1]))" $1)
}

# Check if a local.yaml file is available in the current directory and
# use that as the hashdist profile
if [ -r local.yaml ]; then
    HASHDIST_PROFILE=$(realpath local.yaml)
fi

# Check if a profile is given at the command line
if [ $# -ge 1 ]
then
    HASHDIST_PROFILE=$(realpath $1)
fi

# Check if a profile is given as an environment variable
if [ ! -z ${FENICS_INSTALL_HASHDIST_PROFILE} ]
then
    HASHDIST_PROFILE=$(realpath ${FENICS_INSTALL_HASHDIST_PROFILE})
fi

# Check for wget and curl
if hash wget > /dev/null 2>&1; then
    FETCH="wget -qO-"
elif hash curl > /dev/null 2>&1; then
    FETCH="curl -s"
else
    echo >&2 "This script requires either curl or wget to run."
    exit 1
fi

# Get number of processes to use for build
: ${PROCS:=4}

# Use host Python in the profile if set to 1
: ${FENICS_INSTALL_USE_HOST_PYTHON:=0}

# More verbose output when building
: ${VERBOSE:=0}

# Create a temporary directory for downloads
ODIR=$(pwd)
DIR=$(mktemp -d /tmp/fenics-install.XXXXXX)
echo "Created temporary directory $DIR for FEniCS installation."
echo
cd $DIR

# Download HashDist
echo "Downloading HashDist..."
git clone https://github.com/hippylib/hashdist.git
echo ""

# Download HashStack
echo "Downloading HashStack..."
git clone https://github.com/hippylib/hashstack.git
echo ""

# Download FEniCS Install
echo "Downloading FEniCS Install..."
git clone https://bitbucket.org/fenics-project/fenics-developer-tools.git
echo ""

# Run hit init-home to initialize hashdist directory if it does not exist and
# add a source cache mirror to fenicsproject.org
if [ ! -d $HOME/.hashdist ]; then
    ./hashdist/bin/hit init-home
    # FIXME: consider adding this back when issue #353 has been fixed in hashdist
    #sed -i.bak 's| - dir: ./src| - dir: ./src\'$'\n - url: http://fenicsproject.org/pub/hashdist/src/|' $HOME/.hashdist/config.yaml
fi

if ( [ -z $FENICS_INSTALL_BUILD_TYPE  ] &&  [ -z ${HASHDIST_PROFILE} ] ); then
    # Ask for building stable/development version or only dependencies
    echo "Ready to build FEniCS. This may take about 2 hours."
    echo ""
    echo "The following build types are supported:"
    echo ""
    echo "  [0] development version of FEniCS [default]"
    echo "  [1] latest stable version of FEniCS, currently 2016.2"
    echo "  [2] only dependencies, including e.g. MPI, PETSc, Swig, Boost"
    echo ""
    echo "Please select build type [0, 1, 2]: "
    read  BUILD_TYPE < /dev/tty
else
    BUILD_TYPE=$FENICS_INSTALL_BUILD_TYPE
fi

# Select profile
if   [ "x$OS" = "xLinux" ]; then
    PROFILE="fenics.Linux.yaml"
elif [ "x$OS" = "xDarwin" ]; then
    PROFILE="fenics.Darwin.yaml"
    NO_CERT="--no-check-certificate"
elif [ "x$OS" = "xCYGWIN_NT-6.1" ]; then
    PROFILE="fenics.Cygwin.yaml"
else
    echo "*** Unknown operating system: $OS"
    exit 1
fi
if [ "x$BUILD_TYPE" = "x2" ]; then
    PROFILE=fenics-deps$(echo $PROFILE | sed 's/fenics//')
fi
cd hashstack
if [ -z ${HASHDIST_PROFILE} ]; then
    HASHDIST_PROFILE=$(realpath ../fenics-developer-tools/install/profiles/$PROFILE)
else
    BUILD_TYPE="custom"
fi
cp ${HASHDIST_PROFILE} default.yaml
echo "Using HashDist profile ${HASHDIST_PROFILE}."
echo ""

# Use latest changesets from master if development build is requested
FENICS_CHANGESETS=""
if ( [ "x$BUILD_TYPE" = "x0" ] || [ -z $BUILD_TYPE ] ); then
    BUILD_TYPE="development"
    CONFIG_FILE="fenics.dev"
    echo "Development build requested."
    echo ""
    PACKAGES="dijitso dolfin ffc fiat instant mshr ufl"
    for PACKAGE in $PACKAGES; do
        CHANGESET=$(git ls-remote https://bitbucket.org/fenics-project/$PACKAGE.git HEAD | awk '{ print $1 }')
        FENICS_CHANGESETS="$FENICS_CHANGESETS # Changeset for $PACKAGE: $CHANGESET"$'\n'
	HIT_FETCH_OUTPUT=$(../hashdist/bin/hit fetch https://bitbucket.org/fenics-project/$PACKAGE/get/$CHANGESET.tar.gz)
	KEY=$(echo $HIT_FETCH_OUTPUT | sed 's/.*- key: \(.*\) url:.*/\1/')
        PACKAGE_FILE_MODIFIED="$PACKAGE.tmp"
        echo "Updating $PACKAGE to changeset $CHANGESET."
        rm -f default.yaml.tmp
        sed "/$PACKAGE:/a\\
\\    sources:\\
\\      - key: $KEY\\
\\        url: https://bitbucket.org/fenics-project/$PACKAGE/get/$CHANGESET.tar.gz\\
" default.yaml > default.yaml.tmp
        mv default.yaml.tmp default.yaml
    done
elif [ "x$BUILD_TYPE" = "x1" ]; then
    BUILD_TYPE="stable"
    CONFIG_FILE="fenics.stable"
    echo "Stable build requested."
elif [ "x$BUILD_TYPE" = "x2" ]; then
    BUILD_TYPE="dependencies"
    CONFIG_FILE="fenics.deps"
    echo "Building dependencies only."
elif [ "x${BUILD_TYPE}" = "xcustom" ]; then
    CONFIG_FILE="fenics.custom"
    echo "Custom build requested."
else
    echo "*** Unknown build type: $BUILD_TYPE"
    exit 1
fi
echo ""

# Use host Python in the profile if FENICS_INSTALL_USE_HOST_PYTHON is set to 1
if [ "x${FENICS_INSTALL_USE_HOST_PYTHON}" = "x1" ]; then
    sed -i.bak 's/link: shared/host: true\'$'\n    use_python_host_packages: true/' default.yaml
fi

# Set arguments for hit build
HIT_BUILD_ARGS="-j${PROCS} ${NO_CERT}"
if [ "x${VERBOSE}" = "x1" ]; then
    HIT_BUILD_ARGS="${HIT_BUILD_ARGS} --verbose"
fi

# Start installation
echo "Starting build..."
echo ""
time -p ../hashdist/bin/hit build ${HIT_BUILD_ARGS}

# Hack needed for Cygwin (https://github.com/hashdist/hashdist/issues/220)
if [ "x$OS" = "xCYGWIN_NT-6.1" ]; then
    chmod +w default/bin
    cp default/bin/python2.7.exe.link default/bin/python2.7.link
fi

# Generate config file for setting variables
PROFILE=default
PROFILE=$(readlink $PROFILE)
PROFILE=$(basename $PROFILE)
HASHDIST_BUILD_STORE=$(grep build_stores: -A1 ~/.hashdist/config.yaml | tail -1 | cut -f2 -d: | tr -d ' ')
if [ "x$HASHDIST_BUILD_STORE" = "x./bld" ]; then
    HASHDIST_BUILD_STORE="$HOME/.hashdist/bld"
fi
cat << EOF > $CONFIG_FILE
# FEniCS configuration file created by fenics-install.sh on $(date)
# Build type: $BUILD_TYPE
$FENICS_CHANGESETS
export PROFILE=$PROFILE
export PROFILE_INSTALL_DIR=$HASHDIST_BUILD_STORE/profile/\$PROFILE
export PATH=\$PROFILE_INSTALL_DIR/bin:\$PATH
export CMAKE_PREFIX_PATH=\$PROFILE_INSTALL_DIR:\$CMAKE_PREFIX_PATH
export PYTHONPATH=\$PROFILE_INSTALL_DIR/lib/python2.7/site-packages:\$PYTHONPATH
export MANPATH=\$PROFILE_INSTALL_DIR/share/man:\$MANPATH
export PKG_CONFIG_PATH=\$PROFILE_INSTALL_DIR/lib/pkgconfig:\$PKG_CONFIG_PATH
export PETSC_DIR=$PROFILE_INSTALL_DIR
export SLEPC_DIR=$PROFILE_INSTALL_DIR
EOF
if [ "x$OS" = "xDarwin" ]; then
    cat << EOF >> $CONFIG_FILE
export PYTHONPATH=\$PROFILE_INSTALL_DIR/Python.framework/Versions/2.7/lib/python2.7/site-packages:\$PYTHONPATH
EOF
fi

if [ "x$BUILD_TYPE" = "xdependencies" ]; then
    cat << EOF >> $CONFIG_FILE
export HDF5_ROOT=\${PROFILE_INSTALL_DIR}
export BOOST_DIR=\${PROFILE_INSTALL_DIR}
export CMAKE_EXTRA_ARGS="-D Boost_USE_MULTITHREADED:BOOL=OFF
-D HDF5_INCLUDE_DIRS=\${PROFILE_INSTALL_DIR}/include
-D HDF5_DIR:PATH=\${PROFILE_INSTALL_DIR}
-D HDF5_C_COMPILER_EXECUTABLE=\${PROFILE_INSTALL_DIR}/bin/h5pcc"
EOF
else
    cat << EOF >> $CONFIG_FILE
export INSTANT_CACHE_DIR=\$HOME/.cache/instant/\$PROFILE
export DIJITSO_CACHE_DIR=\$HOME/.cache/dijitso/\$PROFILE
EOF
fi
cp $CONFIG_FILE $ODIR

# Print information about profile
PROFILE=default
PROFILE=$(readlink $PROFILE)
PROFILE=$(basename $PROFILE)
echo ""
echo "FEniCS successfully installed as profile '$PROFILE'."
echo ""
echo "To use FEniCS, you must add the following to your environment:"
echo ""
cat $CONFIG_FILE | sed -e 's/^/  /'
echo ""
echo "For your convenience, a configuration file named $CONFIG_FILE has been"
echo "created in the current directory. For setting up your environment, you"
echo "should issue the following command:"
echo ""
echo "  source $CONFIG_FILE"
echo ""

