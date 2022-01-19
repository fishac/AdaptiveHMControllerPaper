#!/bin/bash
#Ex: ./sh/runalloptimalitysearch.sh

all_methods="MRIGARKERK33 MRIGARKIRK21a MRIGARKERK45a MRIGARKESDIRK34a"
explicit_methods="MRIGARKERK33 MRIGARKERK45a"
sr_methods="HeunEulerERK BogackiShampineERK DormandPrinceERK"

declare -A problems
problems=(["Bicoupling"]=$all_methods ["Brusselator"]=$all_methods ["Kaps"]=$all_methods ["KPR"]=$all_methods ["ForcedVanderPol"]=$all_methods ["Pleiades"]=$explicit_methods ["FourBody3d"]=$explicit_methods)

declare -A problems_sr

tols="1e-3 1e-5 1e-7"

slow_penalty_factors="10 1000"

# Ensure all relevant optimal data exists.
for problem in ${!problems[@]}; 
do
	for method in ${problems[$problem]}; 
	do
		for tol in $tols; 
		do
			for slow_penalty_factor in $slow_penalty_factors;
			do
				file="./resources/OptimalitySearch/${problem}/${problem}_OptimalitySearch_${method}_${tol}_${slow_penalty_factor}_optimal.csv"
				if [[ ! -f $file ]]; 
				then
					#Ensure Optimality Search exe exists.
					optimality_search_exe_file="./exe/GenericMultirateOptimalitySearchDriver.exe"
					if [[ ! -f ${optimality_search_exe_file} ]]; 
					then
						make GenericMultirateOptimalitySearchDriver.exe
						mv ./GenericMultirateOptimalitySearchDriver.exe ./exe/GenericMultirateOptimalitySearchDriver.exe
					fi
					echo "Running Optimality Search for ${problem} ${method} ${tol} ${slow_penalty_factor}"
					./exe/GenericMultirateOptimalitySearchDriver.exe ${problem} ${method} ${tol} ${slow_penalty_factor} 1e-08 1e-05 0.1 400 10 0.1 > /dev/null
				fi
			done
		done
	done

	for method in $sr_methods;
	do
		for tol in $tols; 
		do
			file="./resources/OptimalitySearch/${problem}/${problem}_OptimalitySearch_${method}_${tol}_optimal.csv"
			if [[ ! -f $file ]]; 
			then
				#Ensure Optimality Search exe exists.
				optimality_search_exe_file="./exe/GenericSingleRateOptimalitySearchDriver.exe"
				if [[ ! -f ${optimality_search_exe_file} ]]; 
				then
					make GenericSingleRateOptimalitySearchDriver.exe
					mv ./GenericSingleRateOptimalitySearchDriver.exe ./exe/GenericSingleRateOptimalitySearchDriver.exe
				fi
				echo "Running Optimality Search for ${problem} ${method} ${tol}"
				./exe/GenericSingleRateOptimalitySearchDriver.exe ${problem} ${method} ${tol} 1e-08 1e-05 0.1 0.1 > /dev/null
			fi
		done
	done
done