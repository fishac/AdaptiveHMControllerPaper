import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

controllers = ['ConstantConstant', 'LinearLinear', 'PIMR', 'PIDMR']
c_mapping = {c: i for i, c in enumerate(controllers)}

measurement_type_order = ['FS', 'SA-mean', 'SA-max', 'LASA-mean', 'LASA-max']
mt_mapping = {mt: i for i, mt in enumerate(measurement_type_order)}
slow_penalty_factors = [100]

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
	
# Slow cost Deviation average vs esf measurement type
def plot2(ct_data):
	temp_df_columns = ["measurement_type","slow_function_evals","slow_function_evals_opt"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df["slow_cost"] = temp_df["slow_function_evals"].div(temp_df["slow_function_evals_opt"])
	plot_df = temp_df.groupby(by="measurement_type",as_index=False).mean()

	key = plot_df['measurement_type'].map(mt_mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="measurement_type",y="slow_cost")
	#sns_plot.set_title("Mean Cost Deviation of measurement_type",fontsize=18)
	sns_plot.set_ylabel("Mean Slow Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=12)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/measurement_tests/plots/esf_slow_cost_mr.png")
	plt.close(fig)
	
# Fast cost Deviation average vs esf measurement type
def plot3(ct_data):
	temp_df_columns = ["measurement_type","fast_function_evals","fast_function_evals_opt"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df["fast_cost"] = temp_df["fast_function_evals"].div(temp_df["fast_function_evals_opt"])
	plot_df = temp_df.groupby(by="measurement_type",as_index=False).mean()

	key = plot_df['measurement_type'].map(mt_mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="measurement_type",y="fast_cost")
	#sns_plot.set_title("Mean Cost Deviation of measurement_type",fontsize=18)
	sns_plot.set_ylabel("Mean Fast Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=12)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/measurement_tests/plots/esf_fast_cost_mr.png")
	plt.close(fig)

# Relative error deviation, by measurement type
def plot4(mt_data):
	temp_df_columns = ["measurement_type","error","tol"]
	temp_df = mt_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	temp_df["err_deviation"] = np.log10(temp_df["error"].div(temp_df["tol"]))

	plot_df = temp_df.groupby(by=["measurement_type"],as_index=False).mean()

	key = plot_df['measurement_type'].map(mt_mapping)
	plot_df = plot_df.iloc[key.argsort()]

	sns_plot = sns.barplot(data=plot_df,x="measurement_type",y="err_deviation",ci=None)
	#sns_plot.set_title("Mean Error Deviation of\nFast Error Measurement Strategy")
	sns_plot.set_xlabel("")
	sns_plot.set_ylabel("Mean Error Deviation",fontsize=14)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.2f",fontsize=12)
	sns_plot.set_ylim(-1,1)
	sns_plot.axhline(color='k',linewidth=1)
	
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/measurement_tests/plots/esf_err_deviation.png")
	plt.close(fig)

# Relative cost average vs controller
def plot5(mt_data):
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
def plot6(mt_data):
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
def plot7(mt_data):
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
def plot8(mt_data):
	temp_df = mt_data[["controller","status"]].copy()
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()
	
	sns_plot = sns.barplot(data=plot_df,x="controller",y="status")
	sns_plot.set_title("failure rate of controllers")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/measurement_tests/plots/controller_failurerate.png")
	plt.close(fig)

# Curviture score vs controller
def plot9(mt_data):
	temp_df_columns = ["controller","mean_2nd_der_h"]

	temp_df = mt_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()
	
	sns_plot = sns.barplot(data=plot_df,x="controller",y="mean_2nd_der_h")
	sns_plot.set_title("curviture score of controllers")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/measurement_tests/plots/controller_curviture.png")
	plt.close(fig)

# Curviture score vs measurement_type
def plot10(mt_data):
	temp_df_columns = ["measurement_type","mean_2nd_der_h"]

	temp_df = mt_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	plot_df = temp_df.groupby(by="measurement_type",as_index=False).mean()
	
	sns_plot = sns.barplot(data=plot_df,x="measurement_type",y="mean_2nd_der_h")
	sns_plot.set_title("curviture score of measurement_type")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/measurement_tests/plots/esf_curviture.png")
	plt.close(fig)
	
# Error Deviation mean vs controller groups, split by problem
def plot11(mt_data):
	temp_df_columns = ["measurement_type","problem","error","tol"]
	temp_df = mt_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	temp_df["err_dev"] = np.log10(temp_df["error"].div(temp_df["tol"]))

	plot_df = temp_df.groupby(by=["measurement_type","problem"],as_index=False).mean()

	key = plot_df['measurement_type'].map(mt_mapping)
	plot_df = plot_df.iloc[key.argsort()]

	sns_plot = sns.barplot(data=plot_df,x="measurement_type",y="err_dev",ci=None,hue="problem")
	#sns_plot.set_title("Mean Error Deviation of Controller",fontsize=18)
	sns_plot.set_xlabel("")
	sns_plot.set_ylabel("Mean Error Deviation",fontsize=14)
	sns_plot.set_ylim(-1,5)
	sns_plot.tick_params(labelsize=12)
	for container in sns_plot.containers:
		sns_plot.bar_label(container,fmt="%.1f",fontsize=8)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/measurement_tests/plots/esf_problem_err_deviation.png")
	plt.close(fig)

def main():
	mt_data = read_processed_measurement_test_data()
	#optimal_data = read_processed_optimal_data()
	if mt_data is not None:
		#plot1(mt_data)
		plot2(mt_data)
		plot3(mt_data)
		plot4(mt_data)
		#plot5(mt_data)
		plot6(mt_data)
		plot7(mt_data)
		plot8(mt_data)
		#plot9(mt_data)
		plot10(mt_data)
		plot11(mt_data)

if __name__ == "__main__":
	main()