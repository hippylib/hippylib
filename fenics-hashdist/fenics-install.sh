#!/usr/bin/env sh
#
# This script installs FEniCS via HashDist.


# Tell script to exit on first error
set -e

# Check operating system
OS=$(uname)

# Don't check certificates
NO_CERT="--no-check-certificate"

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
# Check if a profile is given at the command line
elif [ $# -ge 1 ]; then
    HASHDIST_PROFILE=$(realpath $1)
# Check if a profile is given as an environment variable
elif [ ! -z ${FENICS_INSTALL_HASHDIST_PROFILE} ]; then
    HASHDIST_PROFILE=$(realpath ${FENICS_INSTALL_HASHDIST_PROFILE})
else
    echo "No HASHDIST_PROFILE found. Please specify one."
    exit 1
fi

# Get number of processes to use for build
: ${PROCS:=4}

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

# Run hit init-home to initialize hashdist directory if it does not exist
if [ ! -d $HOME/.hashdist ]; then
    ./hashdist/bin/hit init-home
fi

cd hashstack

BUILD_TYPE="custom"

cp ${HASHDIST_PROFILE} default.yaml
echo "Using HashDist profile ${HASHDIST_PROFILE}."
echo ""

CONFIG_FILE="fenics.custom"
echo "Custom build requested."
echo ""

# Set arguments for hit build
HIT_BUILD_ARGS="-j${PROCS} ${NO_CERT}"
if [ "x${VERBOSE}" = "x1" ]; then
    HIT_BUILD_ARGS="${HIT_BUILD_ARGS} --verbose"
fi

# Start installation
echo "Starting build..."
echo ""
time -p ../hashdist/bin/hit build ${HIT_BUILD_ARGS}

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
cat << EOF >> $CONFIG_FILE
export INSTANT_CACHE_DIR=\$HOME/.cache/instant/\$PROFILE
export DIJITSO_CACHE_DIR=\$HOME/.cache/dijitso/\$PROFILE
EOF

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

