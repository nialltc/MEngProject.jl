using DrWatson
@quickactivate "MEngProject"
using MEngProject, CUDA, DifferentialEquations, PyPlot, NNlib,  ImageFiltering, Images, MEngProject.LaminartKernels, MEngProject.LaminartInitFunc, MEngProject.Utils, BenchmarkTools, Test

using OrdinaryDiffEq, ParameterizedFunctions, LSODA, Sundials, DiffEqDevTools, Noise

batch = 1


files = readdir(datadir("img"))

tspan = (0.0f0, 200f0)

batch_ = string(batch,"_",rand(1000:9999))
mkdir(plotsdir(string("batch",batch_)))
for file in files[2:end]
	
	p = LaminartInitFunc.parameterInit_conv_gpu(datadir("img",file), Parameters.parameters_f32);
	
	u0 = cu(reshape(zeros(Float32, p.dim_i, p.dim_j*(5*2+2)), p.dim_i, p.dim_j, 5*2+2,1))
	
	arr1 = u0[:, :, 1:2,:]
	arr2 = u0[:, :, 1:1,:];

	f = LaminartFunc.LamFunction(
		similar(arr1), #x
		similar(arr1), #m
		similar(arr1), #s
		similar(arr2), #x_lgn,
		similar(arr1), #C,
		similar(arr1), #H_z,
		similar(arr1), # dy_temp,
		similar(arr1), # dm_temp,
		similar(arr1), # dz_temp,
		similar(arr1), # ds_temp,
		similar(arr2), # dv_temp,
		similar(arr1), # H_z_temp,
		similar(arr2), #  V_temp_1,
		similar(arr2), #  V_temp_2,
		similar(arr1), #  Q_temp,
		similar(arr1), #   P_temp
    )

	prob = ODEProblem(f, u0, tspan, p)
# 	@benchmark sol = solve(prob)
	sol = solve(prob)
	
	
	for t ∈ [25,50,100,200]
	#for t ∈ [25,50,100,200,400,800]
# 			for t ∈ [25,50,100]


		axMax = findmax(sol(t)[:,:,:,1])[1]

		for k ∈ 1:2:10
			fig, ax = plt.subplots()
			k2=k+1

			im = ax.imshow(sol(t)[:,:,k,1], cmap=matplotlib.cm.PRGn,
						   vmax=axMax, vmin=-axMax)
			im2 = ax.imshow(sol(t)[:,:,k+1,1], cmap=matplotlib.cm.RdBu_r,
						   vmax=axMax, vmin=-axMax, alpha=0.5)

			cbar = fig.colorbar(im2,  shrink=0.9, ax=ax)
			cbar.ax.set_xlabel("\$k=$k2\$")
					cbar = fig.colorbar(im,  shrink=0.9, ax=ax)
			cbar.ax.set_xlabel("\$k=$k\$")
			layer=Utils.layers[k]
				plt.title("Layer: $layer, \$t=$t\$")
				plt.axis("off")
				fig.tight_layout()
			plt.savefig(plotsdir(string("batch",batch_),string(file,"_",t,"_",Utils.la[k],".png")))
		end

		k=11
		fig, ax = plt.subplots()
		k2=k+1

		im = ax.imshow(sol[:,:,k,1,t], cmap=matplotlib.cm.PRGn,
					   vmax=axMax, vmin=-axMax)
		im2 = ax.imshow(sol[:,:,k+1,1,t], cmap=matplotlib.cm.RdBu_r,
					   vmax=axMax, vmin=-axMax, alpha=0.5)

		cbar = fig.colorbar(im2,  shrink=0.9, ax=ax)
		cbar.ax.set_xlabel("\$v^-\$")
				cbar = fig.colorbar(im,  shrink=0.9, ax=ax)
		cbar.ax.set_xlabel("\$v^+\$")

		layer=Utils.layers[k]
			plt.title("Layer: $layer, \$t=$t\$")
			plt.axis("off")
			fig.tight_layout()

		plt.savefig(plotsdir(string("batch",batch_),string(file,"_",t,"_",Utils.la[k],".png")))
	end
	
	
# time plot
	fig, axs = plt.subplots()
	lines = ["b--","g--","r-.","c-.","r:","c:","m-","y-","m:","y:","b-.","g-."]
	for k ∈ 1:12
		layer=Utils.layers_1[k]
		axs.plot(sol[findmax(sol[:,:,k,1,end])[2][1],findmax(sol[:,:,k,1,end])[2][2],k,1,:],lines[k], label="$layer")
	end
	axs.set_xlabel("Time")
	axs.set_ylabel("Activation")
	plt.legend()
	fig.tight_layout()
	plt.savefig(plotsdir(string("batch",batch_),string(file,"_time.png")))
end
