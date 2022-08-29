import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

controllers = ['I', 'PI', 'PID', 'Gustafsson']
mapping = {controller: i for i, controller in enumerate(controllers)}


def read_processed_controller_test_data():
	controller_test_filename = "./postprocessing/output/controller_tests/data/processed_controller_test_data.csv"
	try:
		data = pd.read_csv(controller_test_filename)
		return data
	except:
		print("No controller test data available.\n")
	return None

# Full cost Deviation average vs controller
def plot1(ct_data):
	temp_df_columns = ["controller","full_function_evals","full_function_evals_opt"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df["full_cost"] = temp_df["full_function_evals"].div(temp_df["full_function_evals_opt"])
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="full_cost")
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=10)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_cost_sr.png")
	plt.close(fig)

# Relative error deviation, by controller
def plot2(ct_data):
	temp_df_columns = ["controller","error","tol"]
	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	temp_df["err_dev"] = np.log10(temp_df["error"].div(temp_df["tol"]))

	plot_df = temp_df.groupby(by=["controller"],as_index=False).mean()

	key = plot_df['controller'].map(mapping)
	plot_df = plot_df.iloc[key.argsort()]

	sns_plot = sns.barplot(data=plot_df,x="controller",y="err_dev",ci=None)
	#sns_plot.set_title("Mean Error Deviation of Controller",fontsize=18)
	sns_plot.set_xlabel("")
	sns_plot.set_ylabel("Mean Error Deviation",fontsize=14)
	sns_plot.set_ylim(0,2)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.2f",fontsize=10)
	sns_plot.tick_params(labelsize=12)
	
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_err_deviation_sr.png")
	plt.close(fig)

# Failure rate vs controller
def plot3(ct_data):
	temp_df = ct_data[["controller","status"]].copy()
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()
	
	sns_plot = sns.barplot(data=plot_df,x="controller",y="status")
	sns_plot.set_title("failure rate of controllers")
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_failurerate_sr.png")
	plt.close(fig)

# Full cost Deviation Median vs controller
def plot4(ct_data):
	temp_df_columns = ["controller","full_function_evals","full_function_evals_opt"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df["full_cost"] = temp_df["full_function_evals"].div(temp_df["full_function_evals_opt"])
	plot_df = temp_df.groupby(by="controller",as_index=False).median()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="full_cost")
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Median Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=10)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_cost_median_sr.png")
	plt.close(fig)

# Full cost Deviation mean vs controller with error bars
def plot5(ct_data):
	temp_df_columns = ["controller","full_function_evals","full_function_evals_opt"]

	temp_df = ct_data[temp_df_columns].copy()
	temp_df["full_cost"] = temp_df["full_function_evals"].div(temp_df["full_function_evals_opt"])
	plot_df = temp_df.groupby(by="controller",as_index=False).mean()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="full_cost",estimator=np.mean)
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=10)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_cost_errbars_sr.png")
	plt.close(fig)
	
# Cost Deviation mean vs controller groups, split by problem
def plot6(ct_data):
	temp_df_columns = ["controller","problem","full_function_evals","full_function_evals_opt"]
	
	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna()
	temp_df["full_cost"] = temp_df["full_function_evals"].div(temp_df["full_function_evals_opt"])
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
		print(temp_df)
	plot_df = temp_df.groupby(by=["controller","problem"],as_index=False).mean()

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="full_cost",hue="problem")
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	
	for container in sns_plot.containers:
		sns_plot.bar_label(container,fmt="%.1f",fontsize=8)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_problem_cost_sr.png")
	plt.close(fig)
	
# Error Deviation mean vs controller groups, split by problem
def plot7(ct_data):
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
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_problem_err_deviation_sr.png")
	plt.close(fig)

def main():
	ct_data = read_processed_controller_test_data()
	print(ct_data)
	print(ct_data['measurement_type'])
	print(ct_data['measurement_type'][0])
	ct_data = ct_data.loc[ct_data['measurement_type'].isnull()]
	print(ct_data)
	if ct_data is not None:
		plot1(ct_data)
		plot2(ct_data)
		plot3(ct_data)
		plot4(ct_data)
		plot5(ct_data)
		plot6(ct_data)
		plot7(ct_data)

if __name__ == "__main__":
	main()