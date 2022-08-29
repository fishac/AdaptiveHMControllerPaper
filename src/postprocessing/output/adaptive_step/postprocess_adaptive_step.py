import numpy as np 
import matplotlib.pyplot as plt

def read_stats_data(problem, method, tol, controllers):
	data = {}
	for controller in controllers:
		#print("\tMethod: " + method, flush=True)
		stats_filename = "./output/" + problem + "/" + problem + "_AdaptiveStep_" + tol + "_" + controller + "_LASA-mean_" + method + "_stats.csv"
		
		#print(stats_filename)
		try:
			stats_data = np.genfromtxt(stats_filename, delimiter=",")

			controller_data = { 
				"total_timesteps": stats_data[0],
				"total_successful_timesteps": stats_data[1],
				"total_microtimesteps": stats_data[2],
				"total_successful_microtimesteps": stats_data[3],
				"rel_err": stats_data[4],
				"abs_err": stats_data[5],
				"fast_function_evals": stats_data[7],
				"slow_function_evals": stats_data[8],
				"implicit_function_evals": stats_data[9],
				"explicit_function_evals": stats_data[10],
				"fast_jacobian_evals": stats_data[12],
				"slow_jacobian_evals": stats_data[13],
				"implicit_jacobian_evals": stats_data[13],
				"status": stats_data[15]
			};

			data[controller] = controller_data
		except:
			print("Problem: (" + problem + ") has no (stats) data from method: (" + method + ")", flush=True)
	return data

def read_tHM_data(problem, method, tol, controllers):
	data = {}
	for controller in controllers:
		#print("\tMethod: " + method, flush=True)
		tHM_filename = "./output/" + problem + "/" + problem + "_AdaptiveStep_" + tol + "_" + controller + "_LASA-mean_" + method + "_SOT.csv"
		#print(tHM_filename)
		try:
			tHM_data = np.genfromtxt(tHM_filename, delimiter=",")
			t = tHM_data[:,0]
			H = tHM_data[:,1]
			M = tHM_data[:,2]
			fast_evals = tHM_data[:,4]
			slow_evals = tHM_data[:,5]

			successful_steps = np.unique(np.array([np.max(np.where(t==i)) for i in t]))
			t_successful = t[successful_steps]
			H_successful = H[successful_steps]
			M_successful = M[successful_steps]
			fast_evals_successful = fast_evals[successful_steps]
			slow_evals_successful = slow_evals[successful_steps]

			controller_data = { "t": t, "t_successful": t_successful, 
			"H": H, "H_successful": H_successful,
			"M": M, "M_successful": M_successful,
			"fast_evals": fast_evals, "slow_evals": slow_evals,
			"fast_evals_successful": fast_evals_successful, "slow_evals_successful":slow_evals_successful }

			data[controller] = controller_data
		except:
			print("Problem: (" + problem + ") has no (tHM) data from method: (" + method + ")", flush=True)
	return data
	
def read_optimal_data(problem, method, tol):
	data = {}
	opt_filename = './resources/OptimalitySearch/' + problem + '/' + problem + '_OptimalitySearch_' + method + '_' + tol + '_10_optimal.csv'
	print(opt_filename)
	try:
		opt_data = np.genfromtxt(opt_filename, delimiter=",")
		data['H'] = opt_data[:,0]
		data['M'] = opt_data[:,1]
		tempH = np.copy(data['H'])
		np.insert(tempH,0,0)
		data['t'] = np.cumsum(tempH)
	except:
		print("Problem: (" + problem + ") has no optimal data from method: (" + method + ")", flush=True)
		return None
	return data

def plot_tHM_successful(problem, method, controller, data):
	fig,ax1 = plt.subplots()
	#ax1.set_title(problem + " Adaptivity with " + method + " using " + controller)
	ax1.set_xlabel("t")
	ax1.set_ylabel("H",color="firebrick")
	ax1.plot(data[controller]["t_successful"][:-1],data[controller]["H_successful"][:-1],'x',color="firebrick")
	ax1.tick_params(axis="y",labelcolor="firebrick")

	ax2 = ax1.twinx()
	ax2.set_ylabel("M",color="dodgerblue")
	ax2.plot(data[controller]["t_successful"][:-1],data[controller]["M_successful"][:-1],'.',color="dodgerblue")
	ax2.tick_params(axis="y",labelcolor="dodgerblue")
	fig.savefig("./postprocessing/output/adaptive_step/plots/" + problem + "_" + method + "_" + controller + "_tHM.png")
	plt.close()

def plot_tHM_successful_line(problem, method, controller, data):
	fig,ax1 = plt.subplots()
	#ax1.set_title(problem + " Adaptivity with " + method + " using " + controller)
	ax1.set_xlabel("t")
	ax1.set_ylabel("H",color="firebrick")
	ax1.plot(data[controller]["t_successful"][:-1],data[controller]["H_successful"][:-1],color="firebrick")
	ax1.tick_params(axis="y",labelcolor="firebrick")

	ax2 = ax1.twinx()
	ax2.set_ylabel("M",color="dodgerblue")
	ax2.plot(data[controller]["t_successful"][:-1],data[controller]["M_successful"][:-1],color="dodgerblue")
	ax2.tick_params(axis="y",labelcolor="dodgerblue")
	fig.savefig("./postprocessing/output/adaptive_step/plots/" + problem + "_" + method + "_" + controller + "_tHM_line.png")
	plt.close()
	
def plot_tHh_successful(problem, method, controller, data):
	fig,ax1 = plt.subplots()
	#ax1.set_title(problem + " Adaptivity with " + method + " using " + controller)
	ax1.set_xlabel("t")
	ax1.plot(data[controller]["t_successful"][:-1],data[controller]["H_successful"][:-1],'x',color="firebrick")
	ax1.tick_params(axis="y",labelcolor="black")

	ax1.plot(data[controller]["t_successful"][:-1],data[controller]["H_successful"][:-1]/data[controller]["M_successful"][:-1],'.',color="dodgerblue")
	ax1.legend(labels=['H','h'])
	fig.savefig("./postprocessing/output/adaptive_step/plots/" + problem + "_" + method + "_" + controller + "_tHh.png")
	plt.close()
	
def plot_tHh_successful_line(problem, method, controller, data):
	fig,ax1 = plt.subplots()
	#ax1.set_title(problem + " Adaptivity with " + method + " using " + controller)
	ax1.set_xlabel("t")
	ax1.plot(data[controller]["t_successful"][:-1],data[controller]["H_successful"][:-1],color="firebrick")
	ax1.tick_params(axis="y",labelcolor="black")

	ax1.plot(data[controller]["t_successful"][:-1],data[controller]["H_successful"][:-1]/data[controller]["M_successful"][:-1],color="dodgerblue")
	ax1.legend(labels=['H','h'])
	fig.savefig("./postprocessing/output/adaptive_step/plots/" + problem + "_" + method + "_" + controller + "_tHh_line.png")
	plt.close()
	
def plot_tHh_successful_line_semilogy(problem, method, controller, data):
	fig,ax1 = plt.subplots()
	#ax1.set_title(problem + " Adaptivity with " + method + " using " + controller)
	ax1.set_xlabel("t",fontsize=14)
	ax1.semilogy(data[controller]["t_successful"][:-1],data[controller]["H_successful"][:-1],color="firebrick")
	ax1.tick_params(axis="y",labelcolor="black",labelsize=14)
	ax1.tick_params(axis="x",labelcolor="black",labelsize=14)

	ax1.semilogy(data[controller]["t_successful"][:-1],data[controller]["H_successful"][:-1]/data[controller]["M_successful"][:-1],color="dodgerblue")
	ax1.legend(labels=['H','h'],fontsize=14)
	fig.savefig("./postprocessing/output/adaptive_step/plots/" + problem + "_" + method + "_" + controller + "_tHh_line_semilogy.png")
	plt.close()
	
def plot_tHM_successful_opt(problem, method, data):
	fig,ax1 = plt.subplots()
	#ax1.set_title(problem + " Adaptivity with " + method)
	ax1.set_xlabel("t")
	ax1.set_ylabel("H",color="firebrick")
	ax1.plot(data["t"],data["H"],'x',color="firebrick")
	ax1.tick_params(axis="y",labelcolor="firebrick")

	ax2 = ax1.twinx()
	ax2.set_ylabel("M",color="dodgerblue")
	ax2.plot(data["t"],data["M"],'.',color="dodgerblue")
	ax2.tick_params(axis="y",labelcolor="dodgerblue")
	fig.savefig("./postprocessing/output/adaptive_step/plots/" + problem + "_" + method + "_tHM_opt.png")
	plt.close()
	
def plot_tHh_successful_opt(problem, method, data):
	fig,ax1 = plt.subplots()
	#ax1.set_title(problem + " Adaptivity with " + method)
	ax1.set_xlabel("t")
	ax1.plot(data["t"],data["H"],'x',color="firebrick")
	ax1.tick_params(axis="y",labelcolor="black")

	ax1.plot(data["t"],data["H"]/data["M"],'.',color="dodgerblue")
	ax1.legend(labels=['H','h'])
	fig.savefig("./postprocessing/output/adaptive_step/plots/" + problem + "_" + method + "_tHh_opt.png")
	plt.close()
	
def plot_tHh_successful_line_opt(problem, method, data):
	fig,ax1 = plt.subplots()
	#ax1.set_title(problem + " Adaptivity with " + method)
	ax1.set_xlabel("t",fontsize=14)
	ax1.plot(data["t"],data["H"],color="firebrick")
	ax1.tick_params(axis="y",labelcolor="black")

	ax1.plot(data["t"],data["H"]/data["M"],color="dodgerblue")
	ax1.legend(labels=['H','h'])
	fig.savefig("./postprocessing/output/adaptive_step/plots/" + problem + "_" + method + "_tHh_line_opt.png")
	plt.close()

def plot_eff_over_time_successful(problem, method, controllers, tol, data):
	fig,ax1 = plt.subplots()
	#ax1.set_title(problem + " Adaptivity Efficiency with " + method)
	ax1.set_xlabel("t")
	ax1.set_ylabel("$cost_n$")
	ax1.tick_params(axis="y",labelcolor="black")
	for controller in controllers:
		combined_cost = 10.0*data[controller]["slow_evals_successful"][1:] + data[controller]["fast_evals_successful"][1:]
		eff = np.divide(data[controller]["H_successful"][1:], combined_cost)
		ax1.plot(data[controller]["t_successful"][1:],eff)
	ax1.legend(controllers)
	fig.savefig("./postprocessing/output/adaptive_step/plots/" + problem + "_" + method + "_" + tol + "_eff.png")
	plt.close()

def plot_eff_over_time_successful_semilogy(problem, method, controllers, tol, data):
	fig,ax1 = plt.subplots()
	#ax1.set_title(problem + " Adaptivity Efficiency with " + method)
	ax1.set_xlabel("t")
	ax1.set_ylabel("$cost_n$")
	ax1.tick_params(axis="y",labelcolor="black")
	for controller in controllers:
		combined_cost = 10.0*data[controller]["slow_evals_successful"][1:] + data[controller]["fast_evals_successful"][1:]
		eff = np.divide(data[controller]["H_successful"][1:], combined_cost)
		ax1.semilogy(data[controller]["t_successful"][1:],eff)
	ax1.legend(controllers)
	fig.savefig("./postprocessing/output/adaptive_step/plots/" + problem + "_" + method + "_" + tol + "_eff_semilogy.png")
	plt.close()
	
def plot_fast_evals_per_tol_by_controller(problem, method, controllers, tols, stats_data_dict):
	tolvals = [float(tol) for tol in tols]
	plot_data = {}
	for controller in controllers:
		plot_data[controller] = []
		
	for tol in tols:
		for controller in controllers:
			plot_data[controller].append(stats_data_dict[tol][controller]["fast_function_evals"])
		
	print(plot_data)
	fig,ax1 = plt.subplots()
	for controller in controllers:
		#if controller == 
		ax1.semilogx(tolvals,plot_data[controller],linestyle='-',marker='.')
	ax1.legend(controllers,fontsize=14)
	plt.gca().invert_xaxis()
	ax1.set_xlabel("TOL",fontsize=14)
	ax1.set_ylabel("$f^{\;f}_{evals}$",fontsize=14)
	ax1.tick_params(labelsize=12)
	plt.tight_layout()
	fig.savefig("./postprocessing/output/adaptive_step/plots/" + problem + "_" + method + "_fastevals.png")
	


def plot_slow_evals_per_tol_by_controller(problem, method, controllers, tols, stats_data_dict):
	tolvals = [float(tol) for tol in tols]
	plot_data = {}
	for controller in controllers:
		plot_data[controller] = []
		
	for tol in tols:
		for controller in controllers:
			plot_data[controller].append(stats_data_dict[tol][controller]["slow_function_evals"])
			
	print(plot_data)
	fig,ax1 = plt.subplots()
	for controller in controllers:
		ax1.semilogx(tolvals,plot_data[controller],linestyle='-',marker='.')
	ax1.legend(controllers,fontsize=14)
	plt.gca().invert_xaxis()
	ax1.set_xlabel("TOL",fontsize=14)
	ax1.set_ylabel("$f^{\;s}_{evals}$",fontsize=14)
	ax1.tick_params(labelsize=12)
	plt.tight_layout()
	fig.savefig("./postprocessing/output/adaptive_step/plots/" + problem + "_" + method + "_slowevals.png")
	
def plot_err_deviation_per_tol_by_controller(problem, method, controllers, tols, stats_data_dict):
	tolvals = [float(tol) for tol in tols]
	plot_data = {}
	for controller in controllers:
		plot_data[controller] = []
		
	for tol in tols:
		for controller in controllers:
			err_dev = np.log10(stats_data_dict[tol][controller]["rel_err"]/float(tol))
			plot_data[controller].append(err_dev)
			
	print(plot_data)
	fig,ax1 = plt.subplots()
	for controller in controllers:
		ax1.semilogx(tolvals,plot_data[controller],linestyle='-',marker='.')
	ax1.legend(controllers,fontsize=14)
	plt.gca().invert_xaxis()
	ax1.set_xlabel("TOL",fontsize=14)
	ax1.set_ylabel("Error Deviation",fontsize=14)
	ax1.tick_params(labelsize=12)
	plt.tight_layout()
	fig.savefig("./postprocessing/output/adaptive_step/plots/" + problem + "_" + method + "_errdev.png")

"""
def print_prediction_data(problem, methods, data):
	output = problem + " Predictions \n"
	output += "Format:\nMethod\nTrue Positives,False Positives\nFalse Negatives,True Negatives\n\n"
	for method in methods:
		output += method + "\n"
		output += str(data[method]["total_true_positives"]) + "," + str(data[method]["total_false_positives"]) + "\n"
		output += str(data[method]["total_false_negatives"]) + "," + str(data[method]["total_true_negatives"]) + "\n"
		output += "\n"
	file = open("./data/" + problem + "_predictions.csv", "w")
	file.write(output)
	file.close()
"""

def print_function_eval_data(problem, methods, data):
	keys = ["rel_err", "abs_err", "fast_function_evals","slow_function_evals","implicit_function_evals",
	"explicit_function_evals","slow_jacobian_evals","implicit_jacobian_evals"]
	output = "method," + ",".join(keys)

	for method in data.keys():
		list_output = []
		for key in keys:
			list_output.append(str(data[method][key]))
		output += "\n" + method + "," + ",".join(list_output)
	file = open("./data/" + problem + "_function_evals.csv", "w")
	file.write(output)
	file.close()

def print_timestep_data(problem, methods, data):
	keys = ["rel_err", "abs_err", "total_timesteps", "total_successful_timesteps", "total_microtimesteps", 
	"total_successful_microtimesteps"]
	output = "method," + ",".join(keys)
	
	for method in data.keys():
		list_output = []
		if (data[method]["status"] == 0):
			for key in keys:
				list_output.append(str(data[method][key]))
		else:
			for key in keys:
				list_output.append("FAILURE_" + str(data[method]["status"]))
		output += "\n" + method + "," + ",".join(list_output)
	file = open("./data/" + problem + "_timesteps.csv", "w")
	file.write(output)
	file.close()

def print_all_function_eval_data(all_data):
	keys = ["rel_err", "abs_err", "fast_function_evals","slow_function_evals","implicit_function_evals",
	"explicit_function_evals","slow_jacobian_evals","implicit_jacobian_evals"]

	output = ""
	for problem in all_data.keys():
		output += problem + "\nmethod," + ",".join(keys)

		for method in all_data[problem].keys():
			list_output = []
			if (all_data[problem][method]["status"] == 0):
				for key in keys:
					list_output.append(str(all_data[problem][method][key]))
			else:
				for key in keys:
					list_output.append("FAILURE_" + str(all_data[problem][method]["status"]))
			output += "\n" + method + "," + ",".join(list_output)
		output += "\n\n"
	file = open("./data/function_evals.csv", "w")
	file.write(output)
	file.close()

def print_all_timestep_data(all_data):
	keys = ["rel_err", "abs_err", "total_timesteps", "total_successful_timesteps", "total_microtimesteps", 
	"total_successful_microtimesteps"]

	output = ""
	for problem in all_data.keys():
		output += problem + "\nmethod," + ",".join(keys)
		
		for method in all_data[problem].keys():
			list_output = []
			if (all_data[problem][method]["status"] == 0):
				for key in keys:
					list_output.append(str(all_data[problem][method][key]))
			else:
				for key in keys:
					list_output.append("FAILURE_" + str(all_data[problem][method]["status"]))
			output += "\n" + method + "," + ",".join(list_output)
		output += "\n\n"
	file = open("./data/timesteps.csv", "w")
	file.write(output)
	file.close()

def main():
	problem = "Brusselator1D"
	method = "MRIGARKERK45a"
	tol = "1e-4"
	controllers = ["ConstantConstant", "LinearLinear", "PIMR", "PIDMR", "PI", "PID", "Gustafsson"]

	stats_data = read_stats_data(problem, method, tol, controllers)
	#print_function_eval_data(problem, methods, stats_data)
	#print_timestep_data(problem, methods, stats_data)

	tHM_data = read_tHM_data(problem, method, tol, controllers)
	opt_data = read_optimal_data(problem, method, tol)
	for controller in controllers:
		plot_tHM_successful(problem, method, controller, tHM_data)
		plot_tHM_successful_line(problem, method, controller, tHM_data)
		plot_tHh_successful(problem, method, controller, tHM_data)
		plot_tHh_successful_line(problem, method, controller, tHM_data)
		plot_tHh_successful_line_semilogy(problem, method, controller, tHM_data)
	
	tols = ["1e-3", "1e-4", "1e-5", "1e-6", "1e-7"]
	stats_data_dict = {}
	for itol in tols:
		tolstats_data = read_tHM_data(problem, method, itol, controllers)
		stats_data_dict[itol] = read_stats_data(problem, method, itol, controllers)
		plot_eff_over_time_successful(problem, method, controllers, itol, tolstats_data)
		plot_eff_over_time_successful_semilogy(problem, method, controllers, itol, tolstats_data)
		
	plot_fast_evals_per_tol_by_controller(problem, method, controllers, tols, stats_data_dict)
	plot_slow_evals_per_tol_by_controller(problem, method, controllers, tols, stats_data_dict)
	plot_err_deviation_per_tol_by_controller(problem, method, controllers, tols, stats_data_dict)
		
	if opt_data is not None:
		plot_tHM_successful_opt(problem, method, opt_data)
		plot_tHh_successful_opt(problem, method, opt_data)
		plot_tHh_successful_line_opt(problem, method, opt_data)



if __name__ == "__main__":
	main()