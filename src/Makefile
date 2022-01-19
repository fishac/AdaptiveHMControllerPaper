CXX = g++
MPICXX = mpicxx
CXXFLAGS = -O2 -std=c++11
LIBS = -larmadillo -lm
INCS = -I ./common \
-I ./methods \
-I ./methods/DIRK \
-I ./methods/DIRK/instances \
-I ./methods/MRIGARK \
-I ./methods/MRIGARK/instances \
-I ./problems \
-I ./controllers \
-I ./drivers/OptimalitySearch \
-I ./drivers/MeasurementTests \
-I ./drivers/ControllerTests 

GenericControllerTestsDriver.exe : ./drivers/ControllerTests/GenericControllerTestsDriver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

GenericMeasurementTestsDriver.exe : ./drivers/MeasurementTests/GenericMeasurementTestsDriver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

GenericMultirateOptimalitySearchDriver.exe : ./drivers/OptimalitySearch/GenericMultirateOptimalitySearchDriver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

GenericSingleRateOptimalitySearchDriver.exe : ./drivers/OptimalitySearch/GenericSingleRateOptimalitySearchDriver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

MPIParameterOptimizationDriver.exe : ./drivers/ControllerParameterOptimization/MPIParameterOptimizationDriver.cpp 
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

clean :
	\rm -rf *.txt *.exe ./exe/*.exe *~ *.dSYM ./output/*/*.csv* ./resources/OptimalitySearchh/*/*.exe ./postprocessing/output/*/plots/*.png ./postprocessing/output/*/data/*.csv 
