#!/bin/bash
#Ex: ./sh/runallmeasurementtests.sh

problems="Bicoupling Brusselator Lienard KPR Kaps FourBody3d Pleiades"
tols="1e-3 1e-5 1e-7"

# Ensure all relevant optimal data exists.
for problem in $problems; 
do
	for tol in $tols; 
	do
		#Ensure Measurement Tests exe exists.
		measurement_test_exe_file="./exe/GenericMeasurementTestsDriver.exe"
		if [[ ! -f ${measurement_test_exe_file} ]]; 
		then
			make GenericMeasurementTestsDriver.exe
			mv ./GenericMeasurementTestsDriver.exe ./exe/GenericMeasurementTestsDriver.exe
		fi
		echo $'\n'Running Measurement Tests for ${problem} ${tol}
		./exe/GenericMeasurementTestsDriver.exe ${problem} ${tol}
		done
done
