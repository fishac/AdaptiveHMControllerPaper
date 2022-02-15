import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

controllers = ['ConstantConstant', 'LinearLinear', 'PIMR', 'PIDMR']
c_mapping = {c: i for i, c in enumerate(controllers)}

measurement_type_order = ['FS', 'SA-mean', 'SA-max', 'LASA-mean', 'LASA-max']
mt_mapping = {mt: i for i, mt in enumerate(measurement_type_order)}
slow_penalty_factors = [10]

def read_processed_measurement_test_data():
	measurement_test_filename = "./postprocessing/output/measurement_tests/data/processed_measurement_test_data.csv"
	try:
		data = pd.read_csv(measurement_test_filename)
		return data
	except:
		print("No measurement test data available.\n")
	return None

def read_processed_optimal_data():
	optimal_filename = "./postprocessing/output/measurement_tests/data/processed_optimal_data.csv"
	try:
		data = pd.read_csv(optimal_filename)
		return data
	except:
		print("No processed optimal data available.\n")
	return None

# Cost Deviation average vs esf measurement type
def plot1(mt_data):
	temp_df_columns = ["measurement_type"]
	relative_cost_columns = []
	for slow_penalty_factor in slow_penalty_factors:
		key = "relative_cost" + str(slow_penalty_factor)
		relative_cost_columns.append(key)
	temp_df_columns += relative_cost_columns

	temp_df = mt_data[temp_df_columns].copy()
	temp_df.dropna()
	temp_df["cost"] = temp_df[relative_cost_columns].mean(axis=1)
	temp_df = temp_df[["measurement_type","cost"]]
	plot_df = temp_df.groupby(by="measurement_type",as_index=False).mean()

	key = plot_df['measurement_type'].map(mt_mapping)
	plot_df = plot_df.iloc[key.argsort()]

	sns_plot = sns.barplot(data=plot_df,x="measurement_type",y="cost")
	#sns_plot.set_title("Mean Cost Deviation of Fast Error Measurement Strategy")
	sns_plot.set_xlabel("")
	sns_plot.set_ylabel("Mean Cost Deviation",fontsize=14)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=10)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/measurement_tests/plots/esf_cost.png")
	plt.close(fig)

# Relative error deviation, by measurement type
def plot2(mt_data):
	temp_df_columns = ["measurement_type","error","tol"]
	temp_df = mt_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	temp_df["log10err"] = np.log10(temp_df["error"])
	temp_df["log10tol"] = np.log10(temp_df["tol"])

	plot_df = temp_df.groupby(by=["measurement_type","log10tol"],as_index=False).mean()
	plot_df["err_deviation"] = plot_df["log10err"].div(plot_df["log10tol"])

	key = plot_df['measurement_type'].map(mt_mapping)
	plot_df = plot_df.iloc[key.argsort()]

	sns_plot = sns.barplot(data=plot_df,x="measurement_type",y="err_deviation",ci=None)
	#sns_plot.set_title("Mean Error Deviation of\nFast Error Measurement Strategy")
	sns_plot.set_xlabel("")
	sns_plot.set_ylabel("Mean Error Deviation",fontsize=14)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.2f",fontsize=10)
	sns_plot.set_ylim(0,2)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.2f")
	
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/measurement_tests/plots/esf_err_deviation.png")
	plt.close(fig)

# Relative cost average vs controller
def plot3(mt_data):
	temp_df_columns = ["controller"]
	relative_cost_columns = []
	for slow_penalty_factor in slow_penalty_factors:
		key = "relative_cost" + str(slow_penalty_factor)
		relative_cost_columns.append(key)
	temp_df_columns += relative_cost_columns

	temp_df = mt_data[temp_df_columns].copy()
	temp_df.dropna()
	temp_df["cost"] = temp_df[relative_cost_columns].mean(axis=1)
	temp_df = temp_df[["controller","cost"]]
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()

	#temp_df2 = mt_data[["controller","status"]]
	#info_df = temp_df2.groupby(by="controller",as_index=False).mean()

	plot_df["controller"] = plot_df["controller"].astype(str)# + "\n(" +  info_df["status"].round(decimals=2).astype(str) + ")"
	
	sns_plot = sns.barplot(data=plot_df,x="controller",y="cost")
	sns_plot.set_title("average cost of controllers")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/measurement_tests/plots/controller_cost.png")
	plt.close(fig)

# Relative error deviation, by controller
def plot4(mt_data):
	temp_df_columns = ["controller","error","tol"]
	temp_df = mt_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	temp_df["log10err"] = np.log10(temp_df["error"])
	temp_df["log10tol"] = np.log10(temp_df["tol"])

	plot_df = temp_df.groupby(by=["controller","log10tol"],as_index=False).mean()
	plot_df["err_deviation"] = plot_df["log10err"].div(plot_df["log10tol"])

	key = plot_df['controller'].map(c_mapping)
	plot_df = plot_df.iloc[key.argsort()]

	sns_plot = sns.barplot(data=plot_df,x="controller",y="err_deviation",ci=None)
	#sns_plot.set_title("Mean Error Deviation of Controller")
	sns_plot.set_xlabel("")
	sns_plot.set_ylabel("Mean Error Deviation")
	sns_plot.set_ylim(0,2)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.2f")
	
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_err_deviation.png")
	plt.close(fig)

# Failure rate vs measurement_type
def plot5(mt_data):
	temp_df = mt_data[["measurement_type","status"]].copy()
	plot_df = temp_df.groupby(by="measurement_type",as_index=False).mean()

	plot_df["measurement_type"] = plot_df["measurement_type"].astype(str)
	
	sns_plot = sns.barplot(data=plot_df,x="measurement_type",y="status")
	sns_plot.set_title("failure rate of measurement_types")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/measurement_tests/plots/esf_failurerate.png")
	plt.close(fig)

# Failure rate vs controller
def plot6(mt_data):
	temp_df = mt_data[["controller","status"]].copy()
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()
	
	sns_plot = sns.barplot(data=plot_df,x="controller",y="status")
	sns_plot.set_title("failure rate of controllers")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/measurement_tests/plots/controller_failurerate.png")
	plt.close(fig)

# Failure rate vs controller
def plot7(mt_data):
	temp_df = mt_data[["controller","status"]].copy()
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()
	
	sns_plot = sns.barplot(data=plot_df,x="controller",y="status")
	sns_plot.set_title("failure rate of controllers")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/measurement_tests/plots/controller_failurerate.png")
	plt.close(fig)

def main():
	mt_data = read_processed_measurement_test_data()
	#optimal_data = read_processed_optimal_data()
	if mt_data is not None:
		plot1(mt_data)
		plot2(mt_data)
		plot3(mt_data)
		plot4(mt_data)
		plot5(mt_data)
		plot6(mt_data)
		plot7(mt_data)

if __name__ == "__main__":
	main()