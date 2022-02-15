import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

controllers = ['ConstantConstant', 'LinearLinear', 'PIMR', 'PIDMR']
mapping = {controller: i for i, controller in enumerate(controllers)}

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

# Cost Deviation average vs controller
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

	key = plot_df['controller'].map(mapping)

	sns_plot = sns.barplot(data=plot_df.iloc[key.argsort()],x="controller",y="cost")
	#sns_plot.set_title("Mean Cost Deviation of Controller",fontsize=18)
	sns_plot.set_ylabel("Mean Cost Deviation",fontsize=14)
	sns_plot.set_xlabel("")
	sns_plot.tick_params(labelsize=12)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.1f",fontsize=10)
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_cost.png")
	plt.close(fig)

# Relative error deviation, by controller
def plot2(ct_data):
	temp_df_columns = ["controller","error","tol"]
	temp_df = ct_data[temp_df_columns].copy()
	temp_df.dropna(axis=0)
	temp_df["log10err"] = np.log10(temp_df["error"])
	temp_df["log10tol"] = np.log10(temp_df["tol"])

	plot_df = temp_df.groupby(by=["controller","log10tol"],as_index=False).mean()
	plot_df["err_deviation"] = plot_df["log10err"].div(plot_df["log10tol"])

	key = plot_df['controller'].map(mapping)
	plot_df = plot_df.iloc[key.argsort()]

	sns_plot = sns.barplot(data=plot_df,x="controller",y="err_deviation",ci=None)
	#sns_plot.set_title("Mean Error Deviation of Controller",fontsize=18)
	sns_plot.set_xlabel("")
	sns_plot.set_ylabel("Mean Error Deviation",fontsize=14)
	sns_plot.set_ylim(0,2)
	sns_plot.bar_label(sns_plot.containers[0],fmt="%.2f",fontsize=10)
	sns_plot.tick_params(labelsize=12)
	
	plt.tight_layout()
	fig = sns_plot.get_figure()
	fig.savefig("./postprocessing/output/controller_tests/plots/controller_err_deviation.png")
	plt.close(fig)


def main():
	ct_data = read_processed_controller_test_data()
	#optimal_data = read_processed_optimal_data()
	if ct_data is not None:
		plot1(ct_data)
		plot2(ct_data)

if __name__ == "__main__":
	main()