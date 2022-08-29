import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

problems = ['Bicoupling', 'Brusselator', 'KPR', 'Kaps', 'Pleiades', 'FourBody3d', 'Lienard']
explicit_problems = ['Pleiades', 'FourBody3d']
methods = ['MRIGARKERK22a', 'MRIGARKERK33', 'MRIGARKIRK21a', 'MRIGARKERK45a', 'MRIGARKESDIRK34a']
explicit_methods = ['MRIGARKERK33', 'MRIGARKERK45a', 'MRIGARKERK22a']
tols = ['1e-3', '1e-5', '1e-7']
slow_penalty_factor = '10'

def read_optimal_data():
	optimal_data = pd.DataFrame([],columns=['problem','method','tol','total_steps','mean_H','mean_M'])
	for problem in problems:
		for method in methods:
			for tol in tols:
				if method in explicit_methods or (method not in explicit_methods and problem not in explicit_problems):
					single_optimal_data = read_single_optimal_data(problem, method, tol)
					if single_optimal_data is not None:
						optimal_data = pd.concat([optimal_data,single_optimal_data],ignore_index=True)
	return optimal_data


def read_single_optimal_data(problem, method, tol):
	optimal_folder = "./resources/OptimalitySearch/" + problem + "/"
	optimal_filename = problem + "_OptimalitySearch_" + method + "_" + tol + "_" + slow_penalty_factor + "_optimal.csv"
	optimal_filepath = optimal_folder + optimal_filename
	try:
		optimal_data = np.genfromtxt(optimal_filepath, delimiter=",")
		H_array = optimal_data[:,0]
		M_array = optimal_data[:,1]
		fs_array = optimal_data[:,3]
		ff_array = optimal_data[:,4]
		method_data = pd.DataFrame(
				[[problem,method,tol,H_array.size,np.mean(H_array),np.mean(M_array),np.sum(fs_array),np.sum(ff_array)]],
				columns=['problem','method','tol','total_steps','mean_H','mean_M','fs','ff']
				)

		return method_data				
	except:
		print("No optimal data for (" + problem + "," + method + "," + tol + "," + slow_penalty_factor + ")")

def process_stat1(op_data):
	temp_df_columns = ["method","mean_H"]

	temp_df = op_data[temp_df_columns].copy()
	stat_df = temp_df.groupby(by="method",as_index=False).mean()
	print("Mean H by method")
	print(stat_df)
	
def process_stat2(op_data):
	temp_df_columns = ["method","mean_M"]

	temp_df = op_data[temp_df_columns].copy()
	stat_df = temp_df.groupby(by="method",as_index=False).mean()
	print("Mean M by method")
	print(stat_df)
	
def process_stat3(op_data):
	temp_df_columns = ["method","mean_H"]

	temp_df = op_data[temp_df_columns].copy()
	stat_df = temp_df.groupby(by="method",as_index=False).std()
	print("Std of mean H values per run, by method")
	print(stat_df)
	
def process_stat4(op_data):
	temp_df_columns = ["method","mean_M"]

	temp_df = op_data[temp_df_columns].copy()
	stat_df = temp_df.groupby(by="method",as_index=False).std()
	print("Std of mean H values per run, by method")
	print(stat_df)
	
def process_stat5(op_data):
	temp_df_columns = ["method","total_steps"]

	temp_df = op_data[temp_df_columns].copy()
	stat_df = temp_df.groupby(by="method",as_index=False).mean()
	print("Mean steps per run, by method")
	print(stat_df)
	
def process_stat6(op_data):
	temp_df_columns = ["method","total_steps","tol"]

	temp_df = op_data[temp_df_columns].copy()
	stat_df = temp_df.groupby(by=["method","tol"],as_index=False).mean()
	print("Mean steps per run, by method and tol")
	print(stat_df)
	
def process_stat7(op_data):
	temp_df_columns = ["method","total_steps","tol"]

	temp_df = op_data[temp_df_columns].copy()
	stat_df = temp_df.groupby(by=["method","tol"],as_index=False).std()
	print("Std of steps per run, by method and tol")
	print(stat_df)
	
def process_stat8(op_data):
	temp_df_columns = ["method","total_steps","tol"]

	temp_df = op_data[temp_df_columns].copy()
	stat_df = temp_df.groupby(by=["method","tol"],as_index=False).max()
	print("Max of steps per run, by method and tol")
	print(stat_df)
	
def process_stat9(op_data):
	temp_df_columns = ["method","total_steps","tol"]

	temp_df = op_data[temp_df_columns].copy()
	stat_df = temp_df.groupby(by=["method","tol"],as_index=False).min()
	print("Min of steps per run, by method and tol")
	print(stat_df)

###################

def plot1(op_data):
	fig, axes = plt.subplots(2,3,sharex=True,sharey=True)
	fig.suptitle("Histogram of Total Timesteps per run, by method")
	
	
	erk22adata = op_data.loc[op_data["method"]=="MRIGARKERK22a"]
	sns.histplot(ax=axes[0,0],data=erk22adata,x="total_steps",bins=8)
	axes[0,0].set_title("MRIGARKERK22a")
	
	erk33data = op_data.loc[op_data["method"]=="MRIGARKERK33"]
	sns.histplot(ax=axes[0,1],data=erk33data,x="total_steps",bins=8)
	axes[0,1].set_title("MRIGARKERK33")
	
	erk45adata = op_data.loc[op_data["method"]=="MRIGARKERK45a"]
	sns.histplot(ax=axes[0,2],data=erk45adata,x="total_steps",bins=8)
	axes[0,2].set_title("MRIGARKERK45a")
	
	esdirk34adata = op_data.loc[op_data["method"]=="MRIGARKESDIRK34a"]
	sns.histplot(ax=axes[1,0],data=esdirk34adata,x="total_steps",bins=8)
	axes[1,0].set_title("MRIGARKESDIRK34a")
	
	irk21adata = op_data.loc[op_data["method"]=="MRIGARKIRK21a"]
	sns.histplot(ax=axes[1,1],data=irk21adata,x="total_steps",bins=8)
	axes[1,1].set_title("MRIGARKIRK21a")
	
	
	plt.tight_layout()
	fig.savefig("./postprocessing/output/optimality_search/plots/total_timesteps_hist.png")
	plt.close(fig)
	
def plot2(op_data):
	plot_data = op_data[["problem","fs"]].groupby(by="problem",as_index=False).sum()
	sns_plot = sns.barplot(data=plot_data,x="problem",y="fs")
	
	for container in sns_plot.containers:
		sns_plot.bar_label(container,fmt="%.1f",fontsize=8)
	
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/optimality_search/plots/total_fs_by_problem.png")
	plt.close(fig)
	
def plot3(op_data):
	plot_data = op_data[["problem","ff"]].groupby(by="problem",as_index=False).sum()
	sns_plot = sns.barplot(data=plot_data,x="problem",y="ff")
	
	for container in sns_plot.containers:
		sns_plot.bar_label(container,fmt="%.1f",fontsize=8)
	
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/optimality_search/plots/total_ff_by_problem.png")
	plt.close(fig)
	
def plot4(op_data):
	plot_data = op_data[["problem","fs","ff"]]
	plot_data["cost"] = 10*plot_data["fs"]+plot_data["ff"]
	plot_data = plot_data.groupby(by="problem",as_index=False).sum()
	sns_plot = sns.barplot(data=plot_data,x="problem",y="cost")
	
	for container in sns_plot.containers:
		sns_plot.bar_label(container,fmt="%.1f",fontsize=8)
	
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/optimality_search/plots/total_cost_by_problem.png")
	plt.close(fig)

def plot_HM2(problem, method, tol, optimal_data):
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.set_title("H and M vs. t for " + problem + " problem\nusing " + method + " with tol=" + tol)
	ax1.set_xlabel("t")

	for i in range(len(optimal_data["t"])-1):
		t_segment = [optimal_data["t"][i],optimal_data["t"][i+1]]
		H_segment = [optimal_data["H"][i],optimal_data["H"][i]]
		ax1.plot(t_segment,H_segment,c="firebrick")
	ax1.set_ylabel("H",c="firebrick")
	ax1.tick_params(axis="y",labelcolor="firebrick")

	for i in range(len(optimal_data["t"])-1):
		t_segment = [optimal_data["t"][i],optimal_data["t"][i+1]]
		M_segment = [optimal_data["M"][i],optimal_data["M"][i]]
		ax2.plot(t_segment,M_segment,c="dodgerblue")
	ax2.set_ylabel("M",c="dodgerblue")
	ax2.tick_params(axis="y",labelcolor="dodgerblue")
	
	print("Saving optimal_HM2 plot for " + problem + " problem with " + method)
	fig.savefig("./postprocessing/output/optimality_search/plots/" + problem + "_" + method + "_optimal_HM2.png")
	plt.close(fig)

def plot_eff(problem, method, tol, optimal_data):
	fig = plt.figure()
	plt.title("eff vs. t for " + problem + " problem\nusing " + method + " with tol=" + tol)
	plt.xlabel("t")
	plt.ylabel("eff (cost/H)")
	plt.scatter(optimal_data["t"][:-1],optimal_data["eff"],c="k")
	
	print("Saving optimal_eff plot for " + problem + " problem with " + method)
	fig.savefig("./postprocessing/output/optimality_search/plots/" + problem + "_" + method + "_optimal_eff.png")
	plt.close(fig)

def plot_eff2(problem, method, tol, optimal_data):
	fig = plt.figure()
	plt.title("eff vs. t for " + problem + " problem\nusing " + method + " with tol=" + tol)
	plt.xlabel("t")
	plt.ylabel("eff (cost/H)")
	for i in range(len(optimal_data["t"])-1):
		t_segment = [optimal_data["t"][i],optimal_data["t"][i+1]]
		eff_segment = [optimal_data["eff"][i],optimal_data["eff"][i]]
		plt.plot(t_segment,eff_segment,c="k")
	
	print("Saving optimal_eff2 plot for " + problem + " problem with " + method)
	fig.savefig("./postprocessing/output/optimality_search/plots/" + problem + "_" + method + "_optimal_eff2.png")
	plt.close(fig)

def analyze_eff(problem, method, tol, optimal_data):
	cost = 0
	for i in range(len(optimal_data["t"])-1):
		t_segment = [optimal_data["t"][i],optimal_data["t"][i+1]]
		eff = optimal_data["eff"][i]
		cost += (t_segment[1] - t_segment[0])*eff
	print(problem + " problem using " + method + " with tol=" + tol + " has total cost (eff integral): " + str(cost))
	return cost

def main():
	op_data = read_optimal_data()
	process_stat1(op_data)
	process_stat2(op_data)
	process_stat3(op_data)
	process_stat4(op_data)
	process_stat5(op_data)
	process_stat6(op_data)
	#process_stat7(op_data)
	process_stat8(op_data)
	process_stat9(op_data)
	
	
	plot1(op_data)
	plot2(op_data)
	plot3(op_data)
	plot4(op_data)

if __name__ == "__main__":
	main()