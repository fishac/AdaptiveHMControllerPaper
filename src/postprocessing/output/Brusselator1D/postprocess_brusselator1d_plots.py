import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def read_data():
	filename = './output/Brusselator1D/Brusselator1D_fixed_t_Y_fsY_ffY.csv'
	raw_data = np.genfromtxt(filename,delimiter=",")
	data = {}
	data["t"] = np.transpose(raw_data[:,0])
	n = int(raw_data[:,1:].shape[1]/9)
	i = 1 
	data["u"] = raw_data[:,i:(i+n)]
	i += n
	data["v"] = raw_data[:,i:(i+n)]
	i += n
	data["w"] = raw_data[:,i:(i+n)]
	i += n
	data["fsu"] = raw_data[:,i:(i+n)]
	i += n
	data["fsv"] = raw_data[:,i:(i+n)]
	i += n
	data["fsw"] = raw_data[:,i:(i+n)]
	i += n
	data["ffu"] = raw_data[:,i:(i+n)]
	i += n
	data["ffv"] = raw_data[:,i:(i+n)]
	i += n
	data["ffw"] = raw_data[:,i:]
	return data

def plot_means_uvw(dataset):
	t = dataset["t"]#[0:250]
	umeans = np.mean(dataset["u"],axis=1)#[0:250]
	vmeans = np.mean(dataset["v"],axis=1)#[0:250]
	wmeans = np.mean(dataset["w"],axis=1)#[0:250]

	plt.figure()
	plt.plot(t,umeans)
	plt.plot(t,vmeans)
	plt.plot(t,wmeans)
	plt.legend(["u","v","w"])
	plt.title("Mean of components over time")
	plt.xlabel("t")
	plt.savefig("./postprocessing/output/Brusselator1D/plots/uvwmeans.png")
	plt.close()

def plot_means_fsuvw(dataset):
	t = dataset["t"]#[0:250]
	umeans = np.mean(dataset["fsu"],axis=1)#[0:250]
	vmeans = np.mean(dataset["fsv"],axis=1)#[0:250]
	wmeans = np.mean(dataset["fsw"],axis=1)#[0:250]

	plt.figure()
	plt.plot(t,umeans)
	plt.plot(t,vmeans)
	plt.plot(t,wmeans)
	plt.legend(["fs(u)","fs(v)","fs(w)"])
	plt.title("Mean of components over time")
	plt.xlabel("t")
	plt.savefig("./postprocessing/output/Brusselator1D/plots/fsuvwmeans.png")
	plt.close()

def plot_means_ffuvw(dataset):
	t = dataset["t"]#[0:250]
	umeans = np.mean(dataset["ffu"],axis=1)#[0:250]
	vmeans = np.mean(dataset["ffv"],axis=1)#[0:250]
	wmeans = np.mean(dataset["ffw"],axis=1)#[0:250]

	plt.figure()
	plt.plot(t,umeans)
	plt.plot(t,vmeans)
	plt.plot(t,wmeans)
	plt.legend(["ff(u)","ff(v)","ff(w)"])
	plt.title("Mean of components over time")
	plt.xlabel("t")
	plt.savefig("./postprocessing/output/Brusselator1D/plots/ffuvwmeans.png")
	plt.close()

def plot_uvw_movie(dataset):
	t = dataset["t"]
	u = dataset["u"]
	v = dataset["v"]
	w = dataset["w"]
	n = u.shape[1]
	x = np.linspace(0,1,n)
	fig,ax = plt.subplots()
	ax.set_xlim((0,1))
	ax.set_ylim((-3,7))
	lineu, = ax.plot(x,u[0,:],color="blue")
	linev, = ax.plot(x,v[0,:],color="orange")
	linew, = ax.plot(x,w[0,:],color="green")
	ax.legend(["u","v","w"])
	title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="left")

	def update(i):
		lineu.set_data(x,u[i,:])
		linev.set_data(x,v[i,:])
		linew.set_data(x,w[i,:])
		title.set_text("t=" + str(t[i]))
		return lineu,linev,linew,title

	ani = animation.FuncAnimation(fig, update, u.shape[0], 
		interval=int((max(t)-min(t))/t.shape[0]*1000), blit=True)
	ani.save('./postprocessing/output/Brusselator1D/plots/uvwmovie.gif')
		

def main():
	dataset = read_data()
	plot_means_uvw(dataset)
	plot_means_fsuvw(dataset)
	plot_means_ffuvw(dataset)
	plot_uvw_movie(dataset)



if __name__ == "__main__":
	main()