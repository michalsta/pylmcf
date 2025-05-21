pip uninstall pylmcf -y
rm -rf build *.so pylmcf.egg-info
pip install --no-deps -v -e .

