using DrWatson
@quickactivate "MEngProject"
using MEngProject, CUDA, DifferentialEquations, PyPlot, NNlib,  ImageFiltering, Images, MEngProject.LaminartKernels, MEngProject.LaminartInitFunc, MEngProject.Utils, BenchmarkTools, Test

using OrdinaryDiffEq, ParameterizedFunctions, LSODA, Sundials, DiffEqDevTools, Noise

batch = 0001


files = readdir(datadir("img"))

tspan = (0.0f0, 800f0)

batch_ = string(batch,"_",rand(1000:9999))

for f in files
	
	p = LaminartInitFunc.parameterInit_conv_gpu(datadir("img",f), Parameters.parameters_f32);
	
	u0 = cureshape(zeros(Float32, p.dim_i, p.dim_j*(5*2+2)), p.dim_i, p.dim_j, 5*2+2,1))
	
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
		similar(arr1), #  A_temp,
		similar(arr1), #   B_temp
    )

	prob = ODEProblem(f, u0, tspan, p)
	@benchmark sol = solve(prob)
	sol = solve(prob)
	
	
	for t ∈ [25,50,100,200,400,800]
		axMax = findmax(sol(t)[:,:,:,1])[1]

		for k ∈ 1:2:10
			fig, ax = plt.subplots()
			k2=k+1

			im = ax.imshow(sol(t)[:,:,k,1], cmap=matplotlib.cm.PRGn,
						   vmax=axMax, vmin=-axMax)
			im2 = ax.imshow(sol(t)[:,:,k+1,1], cmap=matplotlib.cm.RdBu_r,
						   vmax=axMax, vmin=-axMax, alpha=0.5)
			#     if clbar

			cbar = fig.colorbar(im2,  shrink=0.9, ax=ax)
			cbar.ax.set_xlabel("\$k=$k2\$")
					cbar = fig.colorbar(im,  shrink=0.9, ax=ax)
			cbar.ax.set_xlabel("\$k=$k\$")
			#     end
			layer=Utils.layers[k]
				plt.title("Layer: $layer, \$t=$t\$")
				plt.axis("off")
				fig.tight_layout()
			#     plt.show()
			plt.savefig(plotsdir(string("batch",batch_),string(f,"_",t,"_",Utils.la[k],".png")))
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

		plt.savefig(plotsdir(string("batch",batch_),string(f,"_",t,"_",Utils.la[k],".png")))
	end
	
	
end