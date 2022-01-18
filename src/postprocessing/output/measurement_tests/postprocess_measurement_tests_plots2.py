import numpy as np 
import matplotlib.pyplot as plt

mr_controllers = ['ConstantConstant', 'LinearLinear', 'PIMR', 'PIDMR']
#sr_controllers = ['I', 'Gustafsson']
sr_controllers = []
slow_penalty_factors = [10, 1000]

def read_SOT_run_data(problem,controller,tol,measurement_type,method):
	SOT_filename = "./output/" + problem + "/" + problem + "_MeasurementTests_" + controller + "_" + tol + "_" + measurement_type + "_" + method + "_SOT.csv"
	try:
		SOT_data = np.genfromtxt(SOT_filename, delimiter=",")
		data = {
			"key": problem + "_" + controller + "_" + tol + "_" + measurement_type + "_" + method,
			"ts": SOT_data[:,0],
			"hs": SOT_data[:,1],
			"ms": SOT_data[:,2],
			"step_failure": np.zeros(SOT_data[:,0].shape)
		}
		successful_steps = np.unique(np.array([np.max(np.where(data["ts"]==i)) for i in data["ts"]]))
		data["step_failure"][:] = 1
		data["step_failure"][successful_steps] = 0

		return data
	except:
		print("Problem: (" + problem + ") has no SOT data from method: (" + method + ") using controller: (" + controller + ") with tol: (" + tol + ") and measurement_type: (" + measurement_type +")\n")
	return None

def read_SOT_data_multitol(problem,controller,tols,measurement_type,method):
	data = {}
	for tol in tols:
		run_data = read_SOT_run_data(problem,controller,tol,measurement_type,method)
		data[tol] = run_data
	return data

def read_SOT_data_multicontroller(problem,controllers,tol,measurement_type,method):
	data = {}
	for controller in controllers:
		run_data = read_SOT_run_data(problem,controller,tol,measurement_type,method)
		data[controller] = run_data
	return data

# Step size over time for one run
def plot1(mt_data):
	success_ts = mt_data["ts"][mt_data["step_failure"] == 0]
	success_hs = mt_data["hs"][mt_data["step_failure"] == 0]
	failure_ts = mt_data["ts"][mt_data["step_failure"] == 1]
	failure_hs = mt_data["hs"][mt_data["step_failure"] == 1]

	fig = plt.figure()
	plt.plot(success_ts,success_hs)
	plt.scatter(failure_ts,failure_hs,marker='x')
	plt.title("H vs t")
	plt.xlabel("t")
	plt.ylabel("H")
	fig.savefig("./postprocessing/output/measurement_tests/plots/" + mt_data["key"] + "_Hvst.png")

# Step size over time stacked by values (tol/controller)
def plot2(mt_multi_data,values,key,label):
	fig,ax = plt.subplots(len(values),1,sharex=True)
	if len(values) > 1:
		for i in range(len(values)):
			mt_data = mt_multi_data[values[i]]

			success_ts = mt_data["ts"][mt_data["step_failure"] == 0]
			success_hs = mt_data["hs"][mt_data["step_failure"] == 0]
			failure_ts = mt_data["ts"][mt_data["step_failure"] == 1]
			failure_hs = mt_data["hs"][mt_data["step_failure"] == 1]

			ax[i].plot(success_ts,success_hs)
			ax[i].scatter(failure_ts,failure_hs,marker='x')
			ax[i].set_title(label + ": " + values[i])
			ax[i].set_ylabel("H")
		ax[len(values)-1].set_xlabel("t")
	else:
		mt_data = mt_multi_data[values[0]]

		success_ts = mt_data["ts"][mt_data["step_failure"] == 0]
		success_hs = mt_data["hs"][mt_data["step_failure"] == 0]
		failure_ts = mt_data["ts"][mt_data["step_failure"] == 1]
		failure_hs = mt_data["hs"][mt_data["step_failure"] == 1]

		ax.plot(success_ts,success_hs)
		ax.scatter(failure_ts,failure_hs,marker='x')
		ax.set_title(label + ": " + values[0])
		ax.set_ylabel("H")
		ax.set_xlabel("t")
	fig.suptitle("H vs t by " + label)
	plt.tight_layout()
	fig.savefig("./postprocessing/output/measurement_tests/plots/" + key + "_Hvstby" + label + ".png")

# Step size and micro step size over time stacked by values (tol/controller)
def plot3(mt_multi_data,values,key,label):
	fig,ax = plt.subplots(len(values),1,sharex=True)
	if len(values) > 1:
		for i in range(len(values)):
			mt_data = mt_multi_data[values[i]]

			success_ts = mt_data["ts"][mt_data["step_failure"] == 0]
			success_hs = mt_data["hs"][mt_data["step_failure"] == 0]
			success_ms = mt_data["ms"][mt_data["step_failure"] == 0]
			failure_ts = mt_data["ts"][mt_data["step_failure"] == 1]
			failure_hs = mt_data["hs"][mt_data["step_failure"] == 1]
			failure_ms = mt_data["ms"][mt_data["step_failure"] == 1]

			success_microhs = success_hs / success_ms
			failure_microhs = failure_hs / failure_ms

			ax[i].plot(success_ts,success_hs,color='b')
			ax[i].scatter(failure_ts,failure_hs,marker='x',color='b')
			ax[i].plot(success_ts,success_microhs,color='r')
			ax[i].scatter(failure_ts,failure_microhs,marker='x',color='r')
			ax[i].set_title(label + ": " + values[i])
		ax[0].legend(['H','h'])
		ax[len(values)-1].set_xlabel("t")
	else:
		mt_data = mt_multi_data[values[0]]

		success_ts = mt_data["ts"][mt_data["step_failure"] == 0]
		success_hs = mt_data["hs"][mt_data["step_failure"] == 0]
		success_ms = mt_data["ms"][mt_data["step_failure"] == 0]
		failure_ts = mt_data["ts"][mt_data["step_failure"] == 1]
		failure_hs = mt_data["hs"][mt_data["step_failure"] == 1]
		failure_ms = mt_data["ms"][mt_data["step_failure"] == 1]

		success_microhs = success_hs / success_ms
		failure_microhs = failure_hs / failure_ms

		ax.plot(success_ts,success_hs,color='b')
		ax.scatter(failure_ts,failure_hs,marker='x',color='b')
		ax.plot(success_ts,success_microhs,color='r')
		ax.scatter(failure_ts,failure_microhs,marker='x',color='r')
		ax.set_title(label + ": " + values[0])


		ax.legend(['H','h'])
		ax.set_xlabel("t")
	fig.suptitle("H vs t by " + label)
	plt.tight_layout()
	fig.savefig("./postprocessing/output/measurement_tests/plots/" + key + "_Hhvstby" + label + ".png")

def main():
	problem = "Kaps"
	controller = "ConstantConstant"
	controllers = ["ConstantConstant"]
	tols = ["1e-3","1e-5","1e-7"]
	tol = "1e-5"
	measurement_type = "LASA-mean"
	method = "MRIGARKERK33"
	multitol_key = problem + "_" + controller + "_" + measurement_type + "_" + method
	multicontroller_key = problem + "_" + tol + "_" + measurement_type + "_" + method

	mt_data = read_SOT_run_data(problem,controller,tol,measurement_type,method)
	mt_multitol_data = read_SOT_data_multitol(problem,controller,tols,measurement_type,method)
	mt_multicontroller_data = read_SOT_data_multicontroller(problem,controllers,tol,measurement_type,method)
	#plot1(mt_data)
	plot2(mt_multitol_data,tols,multitol_key,"tol")
	plot2(mt_multicontroller_data,controllers,multicontroller_key,"controller")
	plot3(mt_multitol_data,tols,multitol_key,"tol")
	plot3(mt_multicontroller_data,controllers,multicontroller_key,"controller")


if __name__ == "__main__":
	main()