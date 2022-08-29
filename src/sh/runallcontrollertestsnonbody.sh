#!/bin/bash
#Ex: ./sh/runallcontrollertests.sh

# Clear old data
rm ./output/*/*_ControllerTests_*.csv

problems="Bicoupling Brusselator Lienard KPR Kaps"
tols="1e-3 1e-5 1e-7"

# Ensure all relevant optimal data exists.
for problem in $problems; 
do
	for tol in $tols; 
	do
		#Ensure Measurement Tests exe exists.
		controller_test_exe_file="./exe/GenericControllerTestsDriver.exe"
		if [[ ! -f ${controller_test_exe_file} ]]; 
		then
			make GenericControllerTestsDriver.exe
			mv ./GenericControllerTestsDriver.exe ./exe/GenericControllerTestsDriver.exe
		fi
		echo $'\n'Running Measurement Tests for ${problem} ${tol}
		./exe/GenericControllerTestsDriver.exe ${problem} ${tol} all
		done
done
