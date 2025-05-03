INCLUDEDIRS=-I/opt/homebrew/Cellar/pybind11/2.13.6_1/include/ -I/opt/homebrew/Cellar/python@3.13/3.13.3/Frameworks/Python.framework/Versions/3.13/include/python3.13
all:
	g++ -std=c++23 -Wall test_compile_all.cpp -shared -o lmcf.so -I.. ${INCLUDEDIRS}
