import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

mr_controllers = ['ConstantConstant', 'LinearLinear', 'PIMR', 'PIDMR']
#sr_controllers = ['I', 'Gustafsson']
sr_controllers = []
slow_penalty_factors = [10]

def read_processed_controller_test_data():
	controller_test_filename = "./postprocessing/output/controller_tests/data/processed_controller_test_data.csv"
	try:
		data = pd.read_csv(controller_test_filename)
		return data
	except:
		print("No controller test data available.\n")
	return None

def read_processed_optimal_data():
	optimal_filename = "./postprocessing/output/controller_tests/data/processed_optimal_data.csv"
	try:
		data = pd.read_csv(optimal_filename)
		return data
	except:
		print("No processed optimal data available.\n")
	return None

# Relative cost average vs controller
def plot1(ct_data):
	temp_df_columns = ["controller"]
	relative_cost_columns = []
	for slow_penalty_factor in slow_penalty_factors:
		key = "relative_cost" + str(slow_penalty_factor)
		relative_cost_columns.append(key)
	temp_df_columns += relative_cost_columns

	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna()
	temp_df["cost"] = temp_df[relative_cost_columns].mean(axis=1)
	temp_df = temp_df[["controller","cost"]]
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()

	#temp_df2 = ct_data[["controller","status"]]
	#info_df = temp_df2.groupby(by="controller",as_index=False).mean()

	plot_df["controller"] = plot_df["controller"].astype(str)# + "\n(" +  info_df["status"].round(decimals=2).astype(str) + ")"
	
	sns_plot = sns.barplot(data=plot_df,x="controller",y="cost")
	sns_plot.set_title("average cost of controllers")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_cost.png")
	plt.close(fig)

# Relative cost average vs tol, by controller
def plot2(ct_data):
	temp_df_columns = ["controller","tol"]
	relative_cost_columns = []
	for slow_penalty_factor in slow_penalty_factors:
		key = "relative_cost" + str(slow_penalty_factor)
		relative_cost_columns.append(key)
	temp_df_columns += relative_cost_columns

	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	temp_df["cost"] = temp_df[relative_cost_columns].mean(axis=1)
	temp_df = temp_df[["controller","cost","tol"]]#.copy()
	plot_df = temp_df.groupby(by=["controller","tol"],as_index=False).mean()

	#Failure rate by controller
	#temp_df2 = ct_data[["controller","status","tol"]]
	#info_df = temp_df2.groupby(by=["controller","tol"],as_index=False).mean()

	sns_plot = sns.lineplot(data=plot_df,x="tol",y="cost",hue="controller",markers=True,dashes=False)
	sns_plot.set_title("average cost of controller at tol")
	sns_plot.set_xscale("log")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_cost_tol.png")
	plt.close(fig)

# Relative error deviation vs tol, by controller
def plot3(ct_data):
	temp_df_columns = ["controller","error","tol"]
	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	temp_df["log10err"] = np.log10(temp_df["error"])
	temp_df["log10tol"] = np.log10(temp_df["tol"])

	plot_df = temp_df.groupby(by=["controller","log10tol"],as_index=False).mean()
	plot_df["err_deviation"] = plot_df["log10err"].div(plot_df["log10tol"])

	sns_plot = sns.lineplot(data=plot_df,x="log10tol",y="err_deviation",hue="controller",markers=True,dashes=False)
	sns_plot.set_title("average err_deviation of controller at tol")
	#sns_plot.set_xscale("log")
	#sns_plot.set_yscale("log")
	sns_plot.set_xlim(-7.1,-2.9)
	sns_plot.set_ylim(0,2)

	#Draw light grey line at err/tol = 1
	unique_tols = np.unique(plot_df["tol"].values)
	plt.axline(xy1=(unique_tols[0],1),xy2=(unique_tols[1],1),color="lightgrey",zorder=1)

	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_err_deviation_tol.png")
	plt.close(fig)

# Absolute error deviation vs tol, by controller
def plot4(ct_data):
	temp_df_columns = ["controller","error","tol"]
	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	temp_df["log10err"] = np.log10(temp_df["error"])
	temp_df["log10tol"] = np.log10(temp_df["tol"])

	plot_df = temp_df.groupby(by=["controller","log10tol"],as_index=False).mean()

	sns_plot = sns.lineplot(data=plot_df,x="log10tol",y="log10err",hue="controller",markers=True,dashes=False)
	sns_plot.set_title("average err of controller at tol")
	#sns_plot.set_xscale("log")
	#sns_plot.set_yscale("log")
	sns_plot.set_xlim(-7.1,-2.9)
	sns_plot.set_ylim(-7.5,-2.5)

	#Draw light grey line at err = tol
	unique_tols = np.unique(plot_df["tol"].values)
	plt.axline(xy1=(unique_tols[0],unique_tols[0]),xy2=(unique_tols[1],unique_tols[1]),color="lightgrey",zorder=1)

	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_err_tol.png")
	plt.close(fig)

# Failure rate vs controller
def plot5(ct_data):
	temp_df = ct_data[["controller","status"]].copy()
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()
	
	sns_plot = sns.barplot(data=plot_df,x="controller",y="status")
	sns_plot.set_title("failure rate of controllers")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_failurerate.png")
	plt.close(fig)

# Curviture score vs controller
def plot6(ct_data):
	temp_df_columns = ["controller","mean_2nd_der_h"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()
	
	sns_plot = sns.barplot(data=plot_df,x="controller",y="mean_2nd_der_h")
	sns_plot.set_title("curviture score of controllers")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_curviture.png")
	plt.close(fig)

def main():
	ct_data = read_processed_controller_test_data()
	#optimal_data = read_processed_optimal_data()
	if ct_data is not None:
		plot1(ct_data)
		plot2(ct_data)
		plot3(ct_data)
		plot4(ct_data)
		plot5(ct_data)
		plot6(ct_data)

if __name__ == "__main__":
	main()