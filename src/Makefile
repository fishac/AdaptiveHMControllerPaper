CXX = g++
MPICXX = mpicxx
CXXFLAGS = -O2 -std=c++11
LIBS = -larmadillo -lm
INCS = -I ./common \
-I ./methods \
-I ./methods/MRGARK \
-I ./methods/MRGARK/Type1Methods \
-I ./methods/MRGARK/Type1Methods/instances \
-I ./methods/DIRK \
-I ./methods/DIRK/instances \
-I ./methods/MRI \
-I ./methods/MRI/instances \
-I ./problems \
-I ./controllers \
-I ./drivers/AdaptiveStep \
-I ./drivers/FixedStep \
-I ./drivers/SingleRate \
-I ./drivers/OrderVerification \
-I ./drivers/OptimalitySearch \
-I ./drivers/MeasurementTests \
-I ./drivers/MeasurementTests2 \
-I ./drivers/ControllerTests 

OrderVerificationDriver.exe : ./drivers/OrderVerification/OrderVerificationDriver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

GenericFixedDriver.exe : ./drivers/FixedStep/GenericFixedDriver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

GenericAdaptiveDriver.exe : ./drivers/AdaptiveStep/GenericAdaptiveDriver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

GenericControllerTestsDriver.exe : ./drivers/ControllerTests/GenericControllerTestsDriver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

GenericMeasurementTestsDriver.exe : ./drivers/MeasurementTests/GenericMeasurementTestsDriver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

GenericMultirateOptimalitySearchDriver.exe : ./drivers/OptimalitySearch/GenericMultirateOptimalitySearchDriver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

GenericSingleRateOptimalitySearchDriver.exe : ./drivers/OptimalitySearch/GenericSingleRateOptimalitySearchDriver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

GenericParameterOptimizationDriver.exe : ./drivers/ControllerParameterOptimization/GenericParameterOptimizationDriver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)
	
GenericRandomTaskDriver.exe : ./drivers/RandomTask/GenericRandomTaskDriver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

OMPParameterOptimizationDriver.exe : ./drivers/ControllerParameterOptimization/OMPParameterOptimizationDriver.cpp 
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $^ $(LIBS) $(INCS)

MPIParameterOptimizationDriver.exe : ./drivers/ControllerParameterOptimization/MPIParameterOptimizationDriver.cpp 
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)
	
MPIParameterOptimizationNoNBodyDriver.exe : ./drivers/ControllerParameterOptimization/MPIParameterOptimizationNoNBodyDriver.cpp 
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)
	
MPIParameterOptimizationDriverAltCost.exe : ./drivers/ControllerParameterOptimization/MPIParameterOptimizationDriverAltCost.cpp 
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)

Brusselator1DFixedDriver.exe : ./drivers/Brusselator1D/Brusselator1DFixedDriver.cpp 
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(INCS)
	
clean :
	\rm -rf *.txt *.exe *~ *.dSYM ./output/*/*.csv ./postprocessing/output/*/plots/*.png ./postprocessing/output/*/data/*.csv 
