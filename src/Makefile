# This Makefile is not meant for end-users but kept here as a convenience for the developers.
# Users should just use the setup.py script provided in parent directory.


INCLUDEDIRS=-I.. -I$(wildcard /opt/homebrew/Cellar/pybind11/*/include/) -I$(wildcard /opt/homebrew/Cellar/python@3.13/*/Frameworks/Python.framework/Versions/3.13/include/python3.13)
CFLAGS=-std=c++20 -Wall
syntax:
	$(CXX) $(CFLAGS) $(INCLUDEDIRS) -fsyntax-only test_compile_all.cpp
lmcf.so:
	$(CXX) $(CFLAGS) $(INCLUDEDIRS) test_compile_all.cpp -shared -o lmcf.so
