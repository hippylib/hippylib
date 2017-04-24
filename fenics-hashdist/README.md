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

## Note on SSL Certificate error

Hashdist automatically connects to the internet to download the
necessaries dependencies.  Some websites
(e.g. https://pypi.python.org/pypi) require a valid SSL Certificate to
download the package you need.

If you encounter an error of this type:

```
[ERROR] urllib failed to download (reason: [SSL:
 CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:590)):

<url_of_the_package_you_are_trying_to_download>


 *[CRITICAL]* You may wish to check your Internet connection or the
 remote server
 ```
 
 it means that hashdist (urllib to be precise) was not able to find a
 valid SSL Certificate to use.
 
 A workaround suggested
 [here](http://stackoverflow.com/questions/25981703/pip-install-fails-with-connection-error-ssl-certificate-verify-failed-certi)
 is to download the cURL certificate from
 `http://curl.haxx.se/ca/cacert.pem` and copy it in your
 `/etc/ssl/certs` folder (note: this requires sudo).
