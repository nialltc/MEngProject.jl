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
			(C_1=1.5f0, C_2=0.075f0, ϕ=2.0f0, Γ=0.2f0, η_p=2.1f0, η_m=1.5f0, λ=1.5f0, ψ=0.5f0), #base
			(C_1=0.0f0, C_2=0.0f0,   ϕ=2.0f0, Γ=0.2f0, η_p=2.1f0, η_m=1.5f0, λ=1.5f0, ψ=0.5f0), #C_1, C_2 down full
			(C_1=1.5f0, C_2=1.0f0,   ϕ=2.0f0, Γ=0.2f0, η_p=2.1f0, η_m=1.5f0, λ=1.5f0, ψ=0.5f0), #C_2 up 
			(C_1=2.5f0, C_2=0.075f0, ϕ=2.0f0, Γ=0.2f0, η_p=2.1f0, η_m=1.5f0, λ=1.5f0, ψ=0.5f0), #C_1 up
			(C_1=1.5f0, C_2=0.075f0, ϕ=0.0f0, Γ=0.2f0, η_p=2.1f0, η_m=1.5f0, λ=1.5f0, ψ=0.5f0), #ϕ down full
			(C_1=1.5f0, C_2=0.075f0, ϕ=4.0f0, Γ=0.2f0, η_p=2.1f0, η_m=1.5f0, λ=1.5f0, ψ=0.5f0), #ϕ up double
			(C_1=1.5f0, C_2=0.075f0, ϕ=2.0f0, Γ=0.1f0, η_p=2.1f0, η_m=1.5f0, λ=1.5f0, ψ=0.5f0), #Γ down half
			(C_1=1.5f0, C_2=0.075f0, ϕ=2.0f0, Γ=0.4f0, η_p=2.1f0, η_m=1.5f0, λ=1.5f0, ψ=0.5f0), #Γ up double
			(C_1=1.5f0, C_2=0.075f0, ϕ=2.0f0, Γ=0.2f0, η_p=3.0f0, η_m=0.5f0, λ=1.5f0, ψ=0.5f0), #η_p up, η_m down
			(C_1=1.5f0, C_2=0.075f0, ϕ=2.0f0, Γ=0.2f0, η_p=1.0f0, η_m=2.5f0, λ=1.5f0, ψ=0.5f0), #η_p down, n_m up
			(C_1=1.5f0, C_2=0.075f0, ϕ=2.0f0, Γ=0.2f0, η_p=2.1f0, η_m=1.5f0, λ=2.5f0, ψ=0.5f0), #λ up
			(C_1=1.5f0, C_2=0.075f0, ϕ=2.0f0, Γ=0.2f0, η_p=2.1f0, η_m=1.5f0, λ=0.5f0, ψ=0.5f0), #λ down
			(C_1=1.5f0, C_2=0.075f0, ϕ=2.0f0, Γ=0.2f0, η_p=2.1f0, η_m=1.5f0, λ=1.5f0, ψ=1.0f0), #ψ up
			(C_1=1.5f0, C_2=0.075f0, ϕ=2.0f0, Γ=0.2f0, η_p=2.1f0, η_m=1.5f0, λ=1.5f0, ψ=0.0f0), #ψ down full
			(C_1=0.5f0, C_2=0.037f0, ϕ=1.0f0, Γ=0.2f0, η_p=2.1f0, η_m=1.5f0, λ=1.5f0, ψ=1.5f0), #all fb down half
			(C_1=0.0f0, C_2=0.0f0,   ϕ=0.0f0, Γ=0.2f0, η_p=2.1f0, η_m=1.5f0, λ=1.5f0, ψ=1.5f0), #all fb down full
			(C_1=3.0f0, C_2=0.125f0, ϕ=4.0f0, Γ=0.2f0, η_p=2.1f0, η_m=1.5f0, λ=1.5f0, ψ=1.5f0), #all fb up double
		]
		test_name =["base","C1C2_0","C2_1","C1_25","phi0","phi4","Gamma01","Gamma04","etaP3_etaM05","etaP1_etaM25","lamda05","lamda15","psi1","psi0","fb_half","fb0","fb_doub"]
		
		test_name_plt =["Base","\$C_1, C_2 =0\$","\$C_2=1\$","\$C_1= 2.5\$","\$\phi =0\$","\$\phi = 4\$","\$\Gamma= 0.101\$","\$\Gamma=0.4\$","\$\eta^+ =3,\eta^-=0.5\$","\$\eta^+=1, \$\eta^-=2.5\$","\$\lamda=0.5\$","\$\lamda=1.5\$","\$\psi = 1\$","\$psi=0\$","All feedback halved","All feedback \$=0\$","All feedback doubled"]
		
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
		# 	@benchmark sol = solve(prob)
			sol = solve(prob)


			for t ∈ [25,50,100,200,400,800]
	# 				for t ∈ [25,50,100]

				v0 = @view sol(t)[:,:,:,1]
				axMax = findmax(v0)[1]

				for k ∈ 1:2:10
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
				end


				k=11
				fig, ax = plt.subplots()
				v1 = @view sol[:,:,k,1,t]
				v2 = @view sol[:,:,k+1,1,t]
				im = ax.imshow(v1, cmap=matplotlib.cm.PRGn,
							   vmax=axMax, vmin=-axMax)
				im2 = ax.imshow(v2, cmap=matplotlib.cm.RdBu_r,
							   vmax=axMax, vmin=-axMax, alpha=0.5)

				cbar = fig.colorbar(im2,  shrink=0.9, ax=ax)
				cbar.ax.set_xlabel("\$v^-\$")
						cbar = fig.colorbar(im,  shrink=0.9, ax=ax)
				cbar.ax.set_xlabel("\$v^+\$")

				layer=Utils.layers[k]
					plt.title(string("Layer: $layer, \$t=$t\$, ", test_name_plt[test_no]))
					plt.axis("off")
					fig.tight_layout()

				plt.savefig(plotsdir(string("batch",batch_),string(file,"_para_",test_name[test_no],"_t",t,"_",Utils.la[k],".png")))
				close("all")
			end


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
			plt.title(string(test_name_plt[test_no])
			plt.legend()
			fig.tight_layout()
			plt.savefig(plotsdir(string("batch",batch_),string(file,"_para_",test_name[test_no],"_time.png")))
			close("all")
		end
	end
end