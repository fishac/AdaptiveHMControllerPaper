import numpy as np 

def read_stats_data(problem, methods, ms):
	data = {}
	for method in methods:
		method_datas = []
		for m in ms:
			stats_filename = "./../../../output/" + problem + "/" + problem + "_Fixed_" + method + "_M" + str(m) + "_stats.csv"
			try:
				stats_data = np.genfromtxt(stats_filename, delimiter=",")

				method_data = { 
					"hs": stats_data[:,0],
					"errs": stats_data[:,1],
					"fast_function_evals": stats_data[:,2],
					"slow_function_evals": stats_data[:,3],
					"implicit_function_evals": stats_data[:,4],
					"explicit_function_evals": stats_data[:,5],
					"fast_jacobian_evals": stats_data[:,6],
					"slow_jacobian_evals": stats_data[:,7],
					"implicit_jacobian_evals": stats_data[:,8],
					"m": m
				};

				method_datas.append(method_data)
			except:
				print("Problem: (" + problem + ") has no data from method: (" + method + ") for m: (" + str(m) + ")\n")
		if method_datas:
			data[method] = method_datas
	return data

def plot_convergence(problem, methods, m, data):
	fig = plt.figure()
	plt.title("H vs. Err for " + problem + " Problem with M=" + str(m))
	legend_items = []
	for method in methods:
		if method in data.keys():
			for method_data in data[method]:
				if method_data["m"] == m:
					plt.loglog(method_data["hs"],method_data["errs"])
					p = np.polyfit(np.log10(method_data["hs"]),method_data["errs"],1)
					legend_items.append(method + " (" + str(np.round(p[0],2)) + ")")
	plt.legend(legend_items)
	fig.savefig("./plots/" + problem + "_M" + str(m) + "_convergence.png")
	fig.close()

def plot_efficiency(problem, method, data):
	fig = plt.figure()
	plt.title("Efficiency for " + problem + " Problem using " + method)
	legend_items = []
	if method in data.keys():
		for method_data in data[method]:
			plt.loglog(method_data["slow_function_evals"],method_data["errs"])
			legend_items.append("M=" + str(method_data["m"]))
	plt.legend(legend_items)
	fig.savefig(legend_items)
	fig.close()

def main():
	problems = ['Brusselator', 'KPR', 'Kaps', 'Pleiades', 'FourBody3d', 'Oregonator', 'Lienard']
	methods = ['EX2_EX2_2_1_A', 'EX4_EX4_3_2_A', 'MRIGARKERK33', 'MRIGARKIRK21a', 'MRIGARKERK45a', 'MRIGARKESDIRK34a']
	ms = [5, 10]

	for problem in problems:
		data = read_stats_data(problem, methods, ms)
		print_function_eval_data(problem, methods, data)
		
		if data:
			for m in ms:
				plot_convergence(problem, methods, m, data)
			for method in methods:
				plot_efficiency(problem, method, data)


if __name__ == "__main__":
	main()