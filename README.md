**Adaptive HM Controllers**

This is the public repository for the paper _Adaptive Time Step Control for Multirate Methods_ by Alex Fish and Daniel Reynolds.

# Requirements

This code base was written in C++11, Python 3, and Matlab.
We used the `g++` compiler for C++ compilation.

C++ was used for the implementation of the numerical methods and controllers, and requires the usage of the [Armadillo](http://arma.sourceforge.net/) library.
Python was used for postprocessing and generating plots, and require the [Matplotlib](https://matplotlib.org/) and [Pandas](https://pandas.pydata.org/) libraries.
Matlab was used for generating "true" solutions at fixed points, solutions accurate to strict tolerances.
Bash was used for simplifying command-line instructions.

If you wish to run the code used to find good parameters for controllers, you will also need some instance of `MPI`. 
For our runs, we used [OpenMPI](https://www.open-mpi.org/).

# Running

### Generating true solutions

True solution data has been uploaded to this repository in the respective `resources/[Problem]` folders.
If you wish to regenerate this data, run the Matlab files in `truesolutions`.

### Fast Error Measurement Strategy Testing

Ensure true solution data exists for the problems being used, in the `resources/[Problem]` folder.
Then run `./sh/runallmeasurementtests.sh`. 
This will test each measurement strategy (FA, SA-mean, SA-max, LASA-mean, LASA-max) over all of the controllers, problems, methods, and tolerances.
This will generate output csv files in the respective `output/[Problem]` folders and output progress to the screen.

### Controller Testing

Ensure true solution data exists for the problems being used, in the `resources/[Problem]` folder.
Then run `./sh/runallcontrollertests.sh`. 
This will test each controller (ConstantConstant, LinearLinear, PIMR, PIDMR) over all of the problems, methods, and tolerances, using one fast error measurement strategy (default LASA-mean).
This will generate output files in the respective `output/[Problem]` folders and output progress to the screen.

### Generating Optimal Data

Run `./sh/runalloptimalitysearch.sh`. 
This will run the Optimal H-M Search Algorithm over all of the problems, methods, and tolerances.
This will generate output files in the respective `output/OptimalitySearch/[Problem]` folders and output progress to the screen.
This will generate a large amount of data and take a while.

### Generating Fast Error Measurement Test Plots

Run

```
python3 ./postprocessing/output/measurement_tests/postprocess_measurement_tests_data.py
python3 ./postprocessing/output/measurement_tests/postprocess_measurement_tests_plots.py
```
This will generate a processed, combined data csv in `./postprocessing/output/measurement_tests/data` and png plots in `./postprocessing/output/measurement_tests/plots`.

### Generating Controller Test Plots


```
python3 ./postprocessing/output/controller_tests/postprocess_controller_tests_data.py
python3 ./postprocessing/output/measurement_tests/postprocess_controller_tests_plots.py
```
This will generate a processed, combined data csv in `./postprocessing/output/controller_tests/data` and png plots in `./postprocessing/output/controller_tests/plots`.

### Finding Good Controller Parameters

Run
```
make MPIParameterOptimizationDriver.exe
mv MPIParameterOptimizationDriver.exe
```
to compile, and 
```
mpiexec -n [N] ./exe/MPIParameterOptimizationDriver.exe [Controller]
```
to run. 
Here, replace `[N]` with your chosen number of MPI ranks (such as `4`, with a minimum of `2`), and `[Controller]` with your chosen controller (such as `PIMR`).
This will print progress to the screen.
Note that this will take hours to days depending on number of MPI ranks used.
Running with the ConstantConstant controller takes about 30 minutes with 36 MPI ranks.
Running with the PIDMR controller takes about 20 hours with 216 MPI ranks.
