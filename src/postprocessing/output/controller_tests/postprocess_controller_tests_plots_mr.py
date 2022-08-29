import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#controllers = ['ConstantConstant', 'LinearLinear', 'PIMR', 'PIDMR']
#controllers = ['I', 'PI', 'PID', 'Gustafsson']
controllers = ['ConstantConstant', 'LinearLinear', 'PIMR', 'PIDMR', 'I', 'PI', 'PID', 'Gustafsson']
mapping = {controller: i for i, controller in enumerate(controllers)}


def read_processed_controller_test_data():
	controller_test_filename = "./postprocessing/output/controller_tests/data/processed_controller_test_data.csv"
	try:
		data = pd.read_csv(controller_test_filename)
		return data
	except:
		print("No controller test data available.\n")
	return None

# Slow cost Deviation average vs controller
def plot1(ct_data):
	temp_df_columns = ["controller","slow_function_evals","slow_function_evals_opt"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df["slow_cost"] = temp_df["slow_function_evals"].div(temp_df["slow_function_evals_opt"])
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()
	plot_df['log_slow_cost'] = np.log10(plot_df['slow_cost'])
	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="slow_cost")#,log=True)
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Slow Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=12)
	sns_plot.set_ylim(0,90)
	#sns_plot.set_yscale("log")
	sns_plot.set_xticklabels(sns_plot.get_xticklabels(),rotation=-40,ha="left")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_slow_cost_mr.png")
	plt.close(fig)
	
# Fast cost Deviation average vs controller
def plot2(ct_data):
	temp_df_columns = ["controller","fast_function_evals","fast_function_evals_opt"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df["fast_cost"] = temp_df["fast_function_evals"].div(temp_df["fast_function_evals_opt"])
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="fast_cost")#,log=True)
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Fast Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	#sns_plot.set_yscale("log")
	sns_plot.tick_params(labelsize=12)
	sns_plot.set_ylim(0,140)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=12)
	sns_plot.set_xticklabels(sns_plot.get_xticklabels(),rotation=-40,ha="left")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_fast_cost_mr.png")
	plt.close(fig)

# Relative error deviation, by controller
def plot3(ct_data):
	temp_df_columns = ["controller","error","tol"]
	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	temp_df["err_dev"] = np.log10(temp_df["error"].div(temp_df["tol"]))

	plot_df = temp_df.groupby(by=["controller"],as_index=False).mean()

	key = plot_df['controller'].map(mapping)
	plot_df = plot_df.iloc[key.argsort()]
	print("Error Deviation Averages")
	print(plot_df)

	sns_plot = sns.barplot(data=plot_df,x="controller",y="err_dev",ci=None)
	#sns_plot.set_title("Mean Error Deviation of Controller",fontsize=18)
	sns_plot.set_xlabel("")
	sns_plot.set_ylabel("Mean Error Deviation",fontsize=14)
	sns_plot.set_ylim(-1,1)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.2f",fontsize=12)
	sns_plot.set_xticklabels(sns_plot.get_xticklabels(),rotation=-40,ha="left")
	sns_plot.tick_params(labelsize=12)
	
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_err_deviation_mr.png")
	plt.close(fig)

# Failure rate vs controller
def plot4(ct_data):
	temp_df = ct_data[["controller","status"]].copy()
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()
	
	sns_plot = sns.barplot(data=plot_df,x="controller",y="status")
	sns_plot.set_title("failure rate of controllers")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_failurerate_mr.png")
	plt.close(fig)

# Slow cost Deviation Median vs controller
def plot5(ct_data):
	temp_df_columns = ["controller","slow_function_evals","slow_function_evals_opt"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df["slow_cost"] = temp_df["slow_function_evals"].div(temp_df["slow_function_evals_opt"])
	plot_df = temp_df.groupby(by="controller",as_index=False).median()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="slow_cost")
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Median Slow Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=10)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_slow_cost_median_mr.png")
	plt.close(fig)

# Fast cost Deviation Median vs controller
def plot6(ct_data):
	temp_df_columns = ["controller","fast_function_evals","fast_function_evals_opt"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df["fast_cost"] = temp_df["fast_function_evals"].div(temp_df["fast_function_evals_opt"])
	plot_df = temp_df.groupby(by="controller",as_index=False).median()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="fast_cost")
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Median Fast Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=10)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_fast_cost_median_mr.png")
	plt.close(fig)
	
# Slow cost Deviation mean vs controller with error bars
def plot7(ct_data):
	temp_df_columns = ["controller","slow_function_evals","slow_function_evals_opt"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df["slow_cost"] = temp_df["slow_function_evals"].div(temp_df["slow_function_evals_opt"])
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="slow_cost",estimator=np.mean)
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Slow Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=10)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_slow_cost_errbars_mr.png")
	plt.close(fig)
	
# Fast cost Deviation mean vs controller with error bars
def plot8(ct_data):
	temp_df_columns = ["controller","fast_function_evals","fast_function_evals_opt"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df["fast_cost"] = temp_df["fast_function_evals"].div(temp_df["fast_function_evals_opt"])
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="fast_cost",estimator=np.mean)
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Fast Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=10)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_fast_cost_errbars_mr.png")
	plt.close(fig)
	
# Cost Deviation mean vs controller groups, split by problem
def plot9(ct_data):
	temp_df_columns = ["controller","problem","slow_function_evals","slow_function_evals_opt"]
	
	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna()
	temp_df["slow_cost"] = temp_df["slow_function_evals"].div(temp_df["slow_function_evals_opt"])
	#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
	#	print(temp_df)
	plot_df = temp_df.groupby(by=["controller","problem"],as_index=False).mean()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="slow_cost",hue="problem")
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Slow Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	
	for container in sns_plot.containers:
		sns_plot.bar_label(container,fmt="%.1f",fontsize=8)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_problem_slow_cost_mr.png")
	plt.close(fig)
	
# Cost Deviation mean vs controller groups, split by problem
def plot10(ct_data):
	temp_df_columns = ["controller","problem","fast_function_evals","fast_function_evals_opt"]
	
	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna()
	temp_df["fast_cost"] = temp_df["fast_function_evals"].div(temp_df["fast_function_evals_opt"])
	#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
	#	print(temp_df)
	plot_df = temp_df.groupby(by=["controller","problem"],as_index=False).mean()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="fast_cost",hue="problem")
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Fast Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	
	for container in sns_plot.containers:
		sns_plot.bar_label(container,fmt="%.1f",fontsize=8)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_problem_fast_cost_mr.png")
	plt.close(fig)
	
# Error Deviation mean vs controller groups, split by problem
def plot11(ct_data):
	temp_df_columns = ["controller","problem","error","tol"]
	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	temp_df["err_dev"] = np.log10(temp_df["error"].div(temp_df["tol"]))

	plot_df = temp_df.groupby(by=["controller","problem"],as_index=False).mean()

	key = plot_df['controller'].map(mapping)
	plot_df = plot_df.iloc[key.argsort()]

	sns_plot = sns.barplot(data=plot_df,x="controller",y="err_dev",ci=None,hue="problem")
	#sns_plot.set_title("Mean Error Deviation of Controller",fontsize=18)
	sns_plot.set_xlabel("")
	sns_plot.set_ylabel("Mean Error Deviation",fontsize=14)
	sns_plot.set_ylim(0,5)
	sns_plot.tick_params(labelsize=12)
	for container in sns_plot.containers:
		sns_plot.bar_label(container,fmt="%.1f",fontsize=8)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_problem_err_deviation_mr.png")
	plt.close(fig)
	
# Slow func evals mean vs controller groups, split by problem
def plot12(ct_data):
	temp_df_columns = ["controller","problem","slow_function_evals"]
	
	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna()
	plot_df = temp_df.groupby(by=["controller","problem"],as_index=False).mean()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="slow_function_evals",hue="problem")
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Slow Function Evaluations",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	
	for container in sns_plot.containers:
		sns_plot.bar_label(container,fmt="%.1f",fontsize=8)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_problem_slow_evals_mr.png")
	plt.close(fig)
	
# Fast func evals mean vs controller groups, split by problem
def plot13(ct_data):
	temp_df_columns = ["controller","problem","fast_function_evals"]
	
	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna()
	plot_df = temp_df.groupby(by=["controller","problem"],as_index=False).mean()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="fast_function_evals",hue="problem")
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Fast Function Evaluations",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	
	for container in sns_plot.containers:
		sns_plot.bar_label(container,fmt="%.1f",fontsize=8)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_problem_fast_evals_mr.png")
	plt.close(fig)

# Failure rate vs controller
def plot14(ct_data):
	temp_df = ct_data[["controller","problem","status"]].copy()
	plot_df = temp_df.groupby(by=["controller","problem"],as_index=False).mean()
	
	sns_plot = sns.barplot(data=plot_df,x="controller",y="status",hue="problem")
	sns_plot.set_title("failure rate of controllers")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_problem_failurerate_mr.png")
	plt.close(fig)

# Combined cost Deviation average vs controller
def plot15(ct_data):
	temp_df_columns = ["controller","slow_function_evals","slow_function_evals_opt","fast_function_evals","fast_function_evals_opt"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df["slow_cost"] = temp_df["slow_function_evals"].div(temp_df["slow_function_evals_opt"])
	temp_df["fast_cost"] = temp_df["fast_function_evals"].div(temp_df["fast_function_evals_opt"])
	temp_df["combined_cost"] = 10.0*temp_df["slow_cost"]+temp_df["fast_cost"]
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="combined_cost")
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Combined Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=10)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_combined_cost_mr.png")
	plt.close(fig)

# Combined cost Deviation average vs controller split by problem
def plot16(ct_data):
	temp_df_columns = ["controller","problem","slow_function_evals","slow_function_evals_opt","fast_function_evals","fast_function_evals_opt"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df["slow_cost"] = temp_df["slow_function_evals"].div(temp_df["slow_function_evals_opt"])
	temp_df["fast_cost"] = temp_df["fast_function_evals"].div(temp_df["fast_function_evals_opt"])
	temp_df["combined_cost"] = 10.0*temp_df["slow_cost"]+temp_df["fast_cost"]
	plot_df = temp_df.groupby(by=["controller","problem"],as_index=False).mean()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="combined_cost",hue="problem")
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Combined Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	for container in sns_plot.containers:
		sns_plot.bar_label(container,fmt="%.1f",fontsize=8)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_problem_combined_cost_mr.png")
	plt.close(fig)

def main():
	ct_data_raw = read_processed_controller_test_data()
	ct_data_mr = ct_data_raw.loc[ct_data_raw['measurement_type'].notnull()]
	ct_data_success = ct_data_mr.loc[ct_data_mr['status']==0]
	if ct_data_raw is not None:
		plot1(ct_data_success)
		plot2(ct_data_success)
		plot3(ct_data_success)
		plot4(ct_data_raw)
		plot5(ct_data_success)
		plot6(ct_data_success)
		plot7(ct_data_success)
		plot8(ct_data_success)
		plot9(ct_data_success)
		plot10(ct_data_success)
		plot11(ct_data_success)
		plot12(ct_data_success)
		plot13(ct_data_success)
		plot14(ct_data_raw)
		plot15(ct_data_success)
		plot16(ct_data_success)

if __name__ == "__main__":
	main()