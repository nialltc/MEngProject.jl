using DrWatson
@quickactivate "MEngProject"
using MEngProject, CUDA, DifferentialEquations, PyPlot, NNlib,  ImageFiltering, Images, MEngProject.LaminartKernels, MEngProject.LaminartInitFunc, MEngProject.Utils, BenchmarkTools, Test

using OrdinaryDiffEq, ParameterizedFunctions, LSODA, Sundials, DiffEqDevTools, Noise

batch = 1


files = readdir(datadir("img", "res_test"))
# files = ["kan_sq_cont_l.png"]

@inbounds begin
	tspan = (0.0f0,100f0)

	batch_ = string(batch,"_",rand(1000:9999))
	mkdir(plotsdir(string("batch",batch_)))
	test_no = 0
	
	test_name = ["025","050","075","100","200","300","400"]
	test_name_plt = ["\$25\times25\$","\$50\times50\$","\$75\times75\$","\$100\times100\$","\$200\times200\$","\$300\times300\$","\$400\times400\$"]
		
	benchm = []
	
	for file in files[1:end]
		

# 		for para_test in para_sets
			test_no += 1
			p = LaminartInitFunc.parameterInit_conv_gpu(datadir("img",file), Parameters.para_var(para_test));

			u0 = cu(reshape(zeros(Float32, p.dim_i, p.dim_j*(5*p.K+2)), p.dim_i, p.dim_j, 5*p.K+2,1))

			arr1 = similar(u0[:, :, 1:2,:])
			arr2 = similar(u0[:, :, 1:1,:])

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
			append!(benchm, @benchmark sol = solve(prob))
			sol = solve(prob)


			t = 100
				v0 = @view sol(t)[:,:,:,1]
				axMax = findmax(v0)[1]


			k = 7
					fig, ax = plt.subplots()

					v1 = @view sol(t)[:,:,k,1]
					v2 = @view sol(t)[:,:,k+1,1]
					im = ax.imshow(v1, cmap=matplotlib.cm.PRGn,
								   vmax=axMax, vmin=-axMax)
					im2 = ax.imshow(v2, cmap=matplotlib.cm.RdBu_r,
								   vmax=axMax, vmin=-axMax, alpha=0.5)

					cbar = fig.colorbar(im2,  shrink=0.9, ax=ax)
					cbar.ax.set_xlabel("\$k=$k2\$")
							cbar = fig.colorbar(im,  shrink=0.9, ax=ax)
					cbar.ax.set_xlabel("\$k=$k\$")
					layer=Utils.layers[k]
						plt.title(string("Layer: $layer, \$t=$t\$, resolution=", test_name_plt[test_no]))
						plt.axis("off")
						fig.tight_layout()
					plt.savefig(plotsdir(string("batch",batch_),string(file,"_res_",test_name[test_no],"_t",t,"_",Utils.la[k],".png")))
					close("all")


		# time plot
		fig, axs = plt.subplots()

		for k ∈ 1:12
			v3 = @view sol[:,:,k,1,end]
			v4 = @view sol[findmax(v3)[2][1],findmax(v3)[2][2],k,1,:]
			layer=Utils.layers_1[k]
			axs.plot(v4,Utils.lines[k], label="$layer")
		end
		axs.set_xlabel("Time")
		axs.set_ylabel("Activation")
		plt.title(test_name_plt[test_no])
		plt.legend()
		fig.tight_layout()
		plt.savefig(plotsdir(string("batch",batch_),string(file,"_para_",test_name[test_no],"_time.png")))
		close("all")
	end
		
		
	# benchmark plot

	fig, ax = plt.subplots()
	for test ∈ 1:test_no
		ax.scatter(median(benchm[test_no].times), benchm[test_no].memory, label=test_name_plt,
				   alpha=0.3, edgecolors="none")
	end
	ax.legend()
	ax.grid(True)
	fig.tight_layout()
	plt.savefig(plotsdir(string("batch",batch_),string(file,"_para_",test_name[test_no],"_time.png")))
	close("all")
		
end