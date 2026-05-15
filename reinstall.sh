pip uninstall pylmcf -y
rm -rf build *.so pylmcf.egg-info
SKBUILD_BUILD_DIR=_skbuild_$(hostname -s) VERBOSE=1 pip install --no-deps -v -e . --no-build-isolation

