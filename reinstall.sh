pip uninstall pylmcf -y
rm -rf build *.so pylmcf.egg-info
VERBOSE=1 pip install --no-deps -v -e .

