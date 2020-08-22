using DrWatson
@quickactivate "MEngProject"
using MEngProject, CUDA, DifferentialEquations, PyPlot, NNlib,  ImageFiltering, Images, MEngProject.LaminartKernels, MEngProject.LaminartInitFunc, MEngProject.Utils, BenchmarkTools, Test

using OrdinaryDiffEq, ParameterizedFunctions, LSODA, Sundials, DiffEqDevTools, Noise

batch = 1


# files = readdir(datadir("img"))
files = ["kan_sq_cont_l.png"]

@inbounds begin
	tspan = (0.0f0,800f0)

	batch_ = string(batch,"_",rand(1000:9999))
	mkdir(plotsdir(string("batch",batch_)))
	test_no = 0
	
	for file in files[1:end]
		para_sets = [
			(H_fact=13.0f0, T_fact=[0.87f0,0.13f0], T_p_m=0.302f0, W_p_same_fact=39f0, W_m_same_fact=330f0, H_l=19, W_l=19), 
(H_fact=13.0f0, T_fact=[0.87f0,0.13f0], T_p_m=0.302f0, W_p_same_fact=39f0, W_m_same_fact=330f0, H_l=15, W_l=19), 
			(H_fact=13.0f0, T_fact=[0.87f0,0.13f0], T_p_m=0.302f0, W_p_same_fact=39f0, W_m_same_fact=330f0, H_l=19, W_l=15), 
						(H_fact=13.0f0, T_fact=[0.87f0,0.13f0], T_p_m=0.302f0, W_p_same_fact=39f0, W_m_same_fact=330f0, H_l=15, W_l=15), 
		]
		test_name = ["base","H_l15","W_l15","H_l15W_l15"]
		test_name_plt = ["Base","Length \$H = 15\$","Length \$W = 15\$","Length \$H, W = 15\$"]
		
		benchm = []
		for para_test in para_sets
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


			t = 800
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
						plt.title(string("Layer: $layer, \$t=$t\$, ", test_name_plt[test_no]))
						plt.axis("off")
						fig.tight_layout()
					plt.savefig(plotsdir(string("batch",batch_),string(file,"_para_",test_name[test_no],"_t",t,"_",Utils.la[k],".png")))
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
		for test in test_name_plt:
			
			ax.scatter(x, y, label=test_name_plt,
					   alpha=0.3, edgecolors='none')

		ax.legend()
		ax.grid(True)
		fig.tight_layout()
		plt.savefig(plotsdir(string("batch",batch_),string(file,"_para_",test_name[test_no],"_time.png")))
		close("all")
		
	end
end