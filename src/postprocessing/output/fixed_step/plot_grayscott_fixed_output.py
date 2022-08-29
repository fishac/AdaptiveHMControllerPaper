import math
import numpy as np 
import matplotlib.pyplot as plt 

def read_stats(problem, methods, Ms):
	data = {}
	for method in methods:
		method_data = {}
		for M in Ms:
			stats_filename = "./../../../output/" + problem + "/" + problem + "_Fixed_" + method + "_M" + str(M) + "_stats.csv"
			stats_data = np.genfromtxt(stats_filename, delimiter=",")

			method_M_data = { 
				"err": stats_data[:,0],
				"H": stats_data[:,1],
				"fast": stats_data[:,2],
				"slow": stats_data[:,3],
				"imp": stats_data[:,4],
				"exp": stats_data[:,5],
				"imp_jac": stats_data[:,6],
			}

			method_data[str(M)] = method_M_data

		data[method] = method_data
	return data

def plot_convergence_method(data, problem, method, Ms):
	plt.figure()
	legend_items = []
	for M in Ms:
		H = data[method][str(M)]["H"]
		err = data[method][str(M)]["err"]
		p = np.polyfit(np.log10(H),np.log10(err),1)
		legend_items.append("M=" + str(M) + " (" + str(np.round(p[0],2)) + ")")
		plt.loglog(H,err)

	plot_reference_orders(data,method,Ms[0],legend_items)

	plt.title(problem + " Convergence using " + method)
	plt.xlabel("H")
	plt.ylabel("Error")
	plt.legend(legend_items)
	plt.savefig("./plots/" + problem + "_" + method + "_convergence.png")
	plt.close()

def plot_convergence_M(data, problem, methods, M):
	plt.figure()
	legend_items = []
	for method in methods:
		H = data[method][str(M)]["H"]
		err = data[method][str(M)]["err"]
		p = np.polyfit(np.log10(H),np.log10(err),1)
		legend_items.append(method + " (" + str(np.round(p[0],2)) + ")")
		plt.loglog(H,err)

	plot_reference_orders(data,methods[0],M,legend_items)

	plt.title(problem + " Convergence using M=" + str(M))
	plt.xlabel("H")
	plt.ylabel("Error")
	plt.legend(legend_items)
	plt.savefig("./plots/" + problem + "_M" + str(M) + "_convergence.png")
	plt.close()

def plot_reference_orders(data,method,M,legend_items):
	H = data[method][str(M)]["H"]

	y_reference = np.max(data[method][str(M)]["err"])
	reference_line = np.power(H,2)*y_reference/(np.max(H)**2)
	plt.loglog(H,reference_line,'--.',color="darkgrey")
	legend_items.append("(2.00)")

	y_reference = np.max(data[method][str(M)]["err"])
	reference_line = np.power(H,3)*y_reference/(np.max(H)**3)
	plt.loglog(H,reference_line,'--x',color="darkgrey")
	legend_items.append("(3.00)")

def main():
	problem = "GrayScott"
	methods = ["EX2_EX2_2_1_A","EX4_EX4_3_2_A"]
	Ms = [5, 10]
	stats = read_stats(problem, methods, Ms)

	for method in methods:
		plot_convergence_method(stats, problem, method, Ms)

	for M in Ms:
		plot_convergence_M(stats, problem, methods, M)

if __name__ == "__main__":
	main()