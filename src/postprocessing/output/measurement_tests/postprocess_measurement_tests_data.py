"""
Processing data to a more usable format
Granularity: entire solves
"""

import numpy as np 

problems = ['Brusselator', 'KPR', 'Kaps', 'Pleiades', 'FourBody3d', 'ForcedVanderPol']
explicit_problems = ['Pleiades', 'FourBody3d']

mr_methods = ['MRIGARKERK33', 'MRIGARKIRK21a', 'MRIGARKERK45a', 'MRIGARKESDIRK34a']
explicit_methods = ['MRIGARKERK33', 'MRIGARKERK45a']
mr_controllers = ['ConstantConstant', 'LinearLinear', 'PIMR', 'PIDMR']

mr_measurement_types = ['FS', 'SA-mean', 'SA-max', 'LASA-mean', 'LASA-max']
#sr_controllers = ['I', 'Gustafsson']
sr_controllers = []
#sr_methods = ['HeunEulerERK', 'BogackiShampineERK', 'DormandPrinceERK']
sr_methods = []
tols = ['1e-3', '1e-5', '1e-7']
slow_penalty_factors = [10]

def read_mr_optimal_data(problem, method, tol, slow_penalty_factor):
	if problem in explicit_problems and method not in explicit_methods:
		return None

	filename = "./resources/OptimalitySearch/" + problem + "/" + problem + "_OptimalitySearch_" + method + "_" + tol + "_" + str(slow_penalty_factor) + "_optimal.csv"
	try:
		search_data = np.genfromtxt(filename, delimiter=",")

		data = {
			#"hs": search_data[:,0],
			#"ms": search_data[:,1],
			#"effs": search_data[:,2],
			"fast_function_evals": np.sum(search_data[:,3]),
			"slow_function_evals": np.sum(search_data[:,4]),
			"implicit_function_evals": np.sum(search_data[:,5]),
			"explicit_function_evals": np.sum(search_data[:,6]),
			"fast_jacobian_evals": np.sum(search_data[:,7]),
			"slow_jacobian_evals": np.sum(search_data[:,8]),
			"implicit_jacobian_evals": np.sum(search_data[:,9])
		}
		return data
	except:
		print("Problem: (" + problem + ") has no optimality search data from method: (" + method + ") with tol: (" + tol + ")\n")
	return None

def read_sr_optimal_data(problem, method, tol):
	filename = "./resources/OptimalitySearch/" + problem + "/" + problem + "_OptimalitySearch_" + method + "_" + tol + "_optimal.csv"
	try:
		search_data = np.genfromtxt(filename, delimiter=",")

		data = {
			#"hs": search_data[:,0],
			#"effs": search_data[:,1],
			"full_function_evals": np.sum(search_data[:,2]),
			"full_jacobian_evals": np.sum(search_data[:,3])
		}
		return data
	except:
		print("Problem: (" + problem + ") has no optimality search data from method: (" + method + ") with tol: (" + tol + ")\n")
	return None

def read_stats_data(problem, method, controller, tol, measurement_type):
	if problem in explicit_problems and method not in explicit_methods:
		return None

	stats_filename = "./output/" + problem + "/" + problem + "_MeasurementTests_" + controller + "_" + tol + "_" + measurement_type + "_" + method + "_stats.csv"
	SOT_filename = "./output/" + problem + "/" + problem + "_MeasurementTests_" + controller + "_" + tol + "_" + measurement_type + "_" + method + "_SOT.csv"
	
	try:
		stats_data = np.genfromtxt(stats_filename, delimiter=",")
		SOT_data = np.genfromtxt(SOT_filename, delimiter=",")

		data = { 
			"hs": SOT_data[:,1],
			"error": stats_data[5],
			"full_function_evals": stats_data[6],
			"fast_function_evals": stats_data[7],
			"slow_function_evals": stats_data[8],
			"implicit_function_evals": stats_data[9],
			"explicit_function_evals": stats_data[10],
			"full_jacobian_evals": stats_data[11],
			"fast_jacobian_evals": stats_data[12],
			"slow_jacobian_evals": stats_data[13],
			"implicit_jacobian_evals": stats_data[14],
			"status": stats_data[15]
		}
		return data
	except:
		print("Problem: (" + problem + ") has no stats data from method: (" + method + ") using controller: (" + controller + ") with tol: (" + tol + ") and measurement_type: (" + measurement_type +")\n")
	return None

def read_mr_data():
	mr_data = {}
	for problem in problems:
		for tol in tols:
			for method in mr_methods:
				for controller in mr_controllers:
					for measurement_type in mr_measurement_types:
						run_data = read_stats_data(problem, method, controller, tol, measurement_type)
						mr_data[problem + '_' + method + '_' + controller + '_' + tol + '_' + measurement_type] = run_data
	return mr_data

def read_sr_data():
	sr_data = {}
	for problem in problems:
		for tol in tols:
			for method in sr_methods:
				for controller in sr_controllers:
					run_data = read_stats_data(problem, method, controller, tol, '0')
					sr_data[problem + '_' + method + '_' + controller + '_' + tol] = run_data
	return sr_data

def read_optimal_data():
	optimal_data = {}
	for problem in problems:
		for tol in tols:
			for method in mr_methods:
				for slow_penalty_factor in slow_penalty_factors:
					optimality_search_run_data = read_mr_optimal_data(problem, method, tol, slow_penalty_factor)
					optimal_data[problem + '_' + method + '_' + tol + '_' + str(slow_penalty_factor)] = optimality_search_run_data

			for method in sr_methods:
				optimality_search_run_data = read_sr_optimal_data(problem, method, tol)
				optimal_data[problem + '_' + method + '_' + tol] = optimality_search_run_data
	return optimal_data

def calc_mean_2nd_der(data,key):
	mean_2nd_der = 0
	npoints = data[key].size
	mean_2nd_der += abs(data[key][0] - 2*data[key][1] + data[key][2])
	mean_2nd_der += abs(data[key][-3] - 2*data[key][-2] + data[key][-1])
	for i in range(1,npoints-1):
		#j = i+1
		mean_2nd_der += abs(data[key][i-1] - 2*data[key][i] + data[key][i+1])
	mean_2nd_der /= npoints
	return mean_2nd_der

def generate_processed_mt_data_line(data,optimality_search_data,header_array,is_mr,problem,method,controller,tol,measurement_type):
	# Set up data
	line_data = {}

	for key in header_array:
		line_data[key] = "nan"

	line_data["problem"] = problem
	line_data["method"] = method
	line_data["controller"] = controller
	line_data["tol"] = tol
	if is_mr:
		line_data["measurement_type"] = measurement_type

	if data["status"] > 0:
		line_data["status"] = 1
	else:
		line_data["status"] = 0

	# If successful run, record its stats
	if data["status"] == 0:
		line_data["error"] = data["error"]
		mean_2nd_der_h = calc_mean_2nd_der(data,"hs")
		line_data["mean_2nd_der_h"] = mean_2nd_der_h

		# If method is multirate, record multirate-related stats
		if is_mr:
			line_data["slow_function_evals"] = data["slow_function_evals"]
			line_data["fast_function_evals"] = data["fast_function_evals"]
			line_data["implicit_function_evals"] = data["implicit_function_evals"]
			line_data["explicit_function_evals"] = data["explicit_function_evals"]
			line_data["slow_jacobian_evals"] = data["slow_jacobian_evals"]
			line_data["fast_jacobian_evals"] = data["fast_jacobian_evals"]
			line_data["implicit_jacobian_evals"] = data["implicit_jacobian_evals"]

			# Record deviation from optimality for a range of slow penalty factors
			for slow_penalty_factor in slow_penalty_factors:
				key = problem + '_' + method + '_' + tol + '_' + str(slow_penalty_factor)
				if optimality_search_data[key]:
					opt_data = optimality_search_data[key]
					line_data['slow_function_evals_opt' + str(slow_penalty_factor)] = opt_data["slow_function_evals"]
					
					# To avoid DIV0 error, only record deviation if optimal data uses that stat
					if opt_data['fast_function_evals'] > 0:
						line_data['fast_function_evals_opt' + str(slow_penalty_factor)] = opt_data['fast_function_evals']
					
					if opt_data['implicit_function_evals'] > 0:
						line_data['implicit_function_evals_opt' + str(slow_penalty_factor)] = opt_data['implicit_function_evals']
					
					if opt_data['explicit_function_evals'] > 0:
						line_data['explicit_function_evals_opt' + str(slow_penalty_factor)] = opt_data['explicit_function_evals']
					
					if opt_data['slow_jacobian_evals'] > 0:
						line_data['slow_jacobian_evals_opt' + str(slow_penalty_factor)] = opt_data['slow_jacobian_evals']
					
					if opt_data['fast_jacobian_evals'] > 0:
						line_data['fast_jacobian_evals_opt' + str(slow_penalty_factor)] = opt_data['fast_jacobian_evals']
					
					if opt_data['implicit_jacobian_evals'] > 0:
						line_data['implicit_jacobian_evals_opt' + str(slow_penalty_factor)] = opt_data['implicit_jacobian_evals']

					if opt_data['slow_function_evals'] > 0 and opt_data['fast_function_evals'] > 0:
						line_data['relative_cost' + str(slow_penalty_factor)] = (slow_penalty_factor*data['slow_function_evals'] + data['fast_function_evals']) / (slow_penalty_factor*opt_data['slow_function_evals'] + opt_data['fast_function_evals'])
					elif opt_data['slow_function_evals'] > 0 and opt_data['implicit_function_evals'] > 0:
						line_data['relative_cost' + str(slow_penalty_factor)] = (slow_penalty_factor*data['slow_function_evals'] + data['implicit_function_evals']) / (slow_penalty_factor*opt_data['slow_function_evals'] + opt_data['implicit_function_evals'])
		# If method is single rate, record single rate-related stats
		else:
			line_data["full_function_evals"] = data["full_function_evals"]
			line_data["full_jacobian_evals"] = data["full_jacobian_evals"]

			# Record deviation from optimality 
			key = problem + '_' + method + '_' + tol
			if optimality_search_data[key]:
				opt_data = optimality_search_data[key]
				line_data['full_function_evals_opt'] = opt_data['full_function_evals']
				
				# To only record deviation if optimal data uses that stat
				if opt_data['full_jacobian_evals'] > 0:
					line_data['full_jacobian_evals_opt'] = opt_data['full_jacobian_evals']

				if opt_data['full_function_evals'] > 0:
					for slow_penalty_factor in slow_penalty_factors:
						line_data['relative_cost' + str(slow_penalty_factor)] = data["full_function_evals"]/opt_data["full_function_evals"]

	line_array = []
	for key in header_array:
		line_array.append(str(line_data[key]))

	line = ",".join(line_array) + "\n"
	return line

def process_measurement_test_data(mr_data, sr_data, optimality_search_data):
	file = open("./postprocessing/output/measurement_tests/data/processed_measurement_test_data.csv", "w")
	header_array = ["problem","method","controller","tol","measurement_type","status","error","mean_2nd_der_h",
	"full_function_evals","slow_function_evals","fast_function_evals","implicit_function_evals","explicit_function_evals",
	"full_jacobian_evals","slow_jacobian_evals","fast_jacobian_evals","implicit_jacobian_evals",
	"full_function_evals_opt","full_jacobian_evals_opt"]
	for slow_penalty_factor in slow_penalty_factors:
		header_array.append("slow_function_evals_opt" + str(slow_penalty_factor))
		header_array.append("fast_function_evals_opt" + str(slow_penalty_factor))
		header_array.append("implicit_function_evals_opt" + str(slow_penalty_factor))
		header_array.append("explicit_function_evals_opt" + str(slow_penalty_factor))
		header_array.append("slow_jacobian_evals_opt" + str(slow_penalty_factor))
		header_array.append("fast_jacobian_evals_opt" + str(slow_penalty_factor))
		header_array.append("implicit_jacobian_evals_opt" + str(slow_penalty_factor))
		header_array.append("relative_cost" + str(slow_penalty_factor))
	header_line = ",".join(header_array) + "\n"
	file.write(header_line)
									
	for problem in problems:
		for tol in tols:
			for method in mr_methods:
				for measurement_type in mr_measurement_types:
					for controller in mr_controllers:
						key = problem + '_' + method + '_' + controller + '_' + tol + '_' + measurement_type
						if mr_data[key]:
							line = generate_processed_mt_data_line(mr_data[key],optimality_search_data,header_array,True,problem,method,controller,tol,measurement_type)
							file.write(line)
			for method in sr_methods:
				for controller in sr_controllers:
					key = problem + '_' + method + '_' + controller + '_' + tol
					if sr_data[key]:
						line = generate_processed_mt_data_line(sr_data[key],optimality_search_data,header_array,False,problem,method,controller,tol,None)
						file.write(line)
	file.close()

def generated_processed_opt_data_line(data,header_array,problem,method,tol,is_mr):
	line_data = {}
	for key in header_array:
		line_data[key] = "nan"

	line_data["problem"] = problem
	line_data["method"] = method
	line_data["tol"] = tol
	if is_mr:
		line_data["slow_function_evals"] = data["slow_function_evals"]
		line_data["fast_function_evals"] = data["fast_function_evals"]
		line_data["implicit_function_evals"] = data["implicit_function_evals"]
		line_data["explicit_function_evals"] = data["explicit_function_evals"]
		line_data["slow_jacobian_evals"] = data["slow_jacobian_evals"]
		line_data["fast_jacobian_evals"] = data["fast_jacobian_evals"]
		line_data["implicit_jacobian_evals"] = data["implicit_jacobian_evals"]
	else:
		line_data["full_function_evals"] = data["full_function_evals"]
		line_data["full_jacobian_evals"] = data["full_jacobian_evals"]

	line_array = []
	for key in header_array:
		line_array.append(str(line_data[key]))

	line = ",".join(line_array) + "\n"
	return line

def process_optimal_data(optimality_search_data):
	file = open("./postprocessing/output/measurement_tests/data/processed_optimal_data.csv", "w")
	header_array = ["problem", "method", "tol","slow_penalty_factor",
	"full_function_evals", "slow_function_evals", "fast_function_evals", "implicit_function_evals", "explicit_function_evals",
	"full_jacobian_evals", "slow_jacobian_evals", "fast_jacobian_evals", "implicit_jacobian_evals"]
	for problem in problems:
		for tol in tols:
			for method in mr_methods:
				for slow_penalty_factor in slow_penalty_factors:
					key = problem + '_' + method + '_' + tol + '_' + str(slow_penalty_factor)
					if optimality_search_data[key]:
						data = optimality_search_data[key]
						line = generated_processed_opt_data_line(data,header_array,problem,method,tol,True)
						file.write(line)
			for method in sr_methods:
					key = problem + '_' + method + '_' + tol
					if optimality_search_data[key]:
						data = optimality_search_data[key]
						line = generated_processed_opt_data_line(data,header_array,problem,method,tol,False)
						file.write(line)
	file.close()

def main():
	mr_data = read_mr_data()
	sr_data = read_sr_data()
	optimality_search_data = read_optimal_data()

	process_measurement_test_data(mr_data, sr_data, optimality_search_data)
	process_optimal_data(optimality_search_data)

if __name__ == "__main__":
	main()