**Adaptive HM Controllers**

This is the public repository for the paper _Adaptive Time Step Control for Multirate Methods_ by Alex Fish and Daniel Reynolds.

# Running and Requirements

All code is assumed to be run from inside the `src` directory.
This code base was written in C++11, Python 3, and Matlab.
The `g++` compiler was used for C++ compilation.

C++ was used for the implementation of the numerical methods and controllers, and requires the usage of the [Armadillo](http://arma.sourceforge.net/) library.
Python was used for postprocessing and generating plots, and require the Matplotlib and Pandas libraries.
Matlab was used for generating "true" solutions at fixed points, solutions accurate to strict tolerances.

