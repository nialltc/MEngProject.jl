using DrWatson
@quickactivate "MEngProject"
using MEngProject, CUDA, DifferentialEquations, PyPlot, NNlib,  ImageFiltering, Images, MEngProject.LaminartKernels, MEngProject.LaminartInitFunc, MEngProject.Utils, BenchmarkTools, Test

using OrdinaryDiffEq, ParameterizedFunctions, LSODA, Sundials, DiffEqDevTools, Noise

batch = 1


files = readdir(datadir("img"))


global benches = []

tspan = (0.0f0,10f0)

batch_ = string(batch,"_",rand(1000:9999))
mkdir(plotsdir(string("bench_imp",batch_)))
file = "kan_sq_cont_l.png"

test_name_plt = ["CPU conv", "GPU conv",  "CPU imfilter", "GPU imfilter FFT", "GPU imfilter FIR", "GPU imfilter IIR"]

# GPU
p = LaminartInitFunc.parameterInit_conv_gpu(datadir("img",file), Parameters.parameters_f32);

u0 = cu(reshape(zeros(Float32, p.dim_i, p.dim_j*(5*p.K+2)), p.dim_i, p.dim_j, 5*p.K+2,1))

arr1 = similar(u0[:, :, 1:2,:])
arr2 = similar(u0[:, :, 1:1,:])

f = LaminartFunc.LamFunction(
	arr1, #x
	similar(arr1), #m
	similar(arr1), #s
	arr2, #x_lgn,
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
bm = @benchmark solve(prob)
push!(benches, bm)



# CPU conv

p = LaminartInitFunc.parameterInit_conv_cpu(datadir("img",file), Parameters.parameters_f32);

u0 = reshape(zeros(Float32, p.dim_i, p.dim_j*(5*p.K+2)), p.dim_i, p.dim_j, 5*p.K+2,1)


f = LaminartFunc.LamFunction(
	arr1, #x
	similar(arr1), #m
	similar(arr1), #s
	arr2, #x_lgn,
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
bm = @benchmark solve(prob)
push!(benches, bm)


# CPU imfilter

p = LaminartInitFunc.parameterInit_imfil_cpu(datadir("img",file), Parameters.parameters_f32);

u0 = reshape(zeros(Float32, p.dim_i, p.dim_j*(5*p.K+2)), p.dim_i, p.dim_j, 5*p.K+2)

arr1 = similar(u0[:, :, 1:2])
arr2 = similar(u0[:, :, 1:1])

f = LaminartFunc.LamFunction_imfil_cpu(
	arr2, #x_lgn,
	arr1, #C,
	similar(arr1), #H_z,
   	similar(arr1), # H_z_temp,
   	similar(arr2), # v_C_temp1,
   	similar(arr2), # v_C_temp2,
   	similar(arr1), # v_C_tempA,
   	similar(arr1[:,:,1]), #W_temp
)
prob = ODEProblem(f, u0, tspan, p)
bm = @benchmark solve(prob)
push!(benches, bm)



# GPU imfilter FFT

p = LaminartInitFunc.parameterInit_imfil_cpu(datadir("img",file), Parameters.parameters_f32);

u0 = reshape(zeros(Float32, p.dim_i, p.dim_j*(5*p.K+2)), p.dim_i, p.dim_j, 5*p.K+2)


f = LaminartFunc.LamFunction_imfil_cpu(
	arr2, #x_lgn,
	arr1, #C,
	similar(arr1), #H_z,
   	similar(arr1), # H_z_temp,
   	similar(arr2), # v_C_temp1,
   	similar(arr2), # v_C_temp2,
   	similar(arr1), # v_C_tempA,
   	similar(arr1[:,:,1]), #W_temp
)
prob = ODEProblem(f, u0, tspan, p)
bm = @benchmark solve(prob)
push!(benches, bm)



# GPU imfilter IIR

p = LaminartInitFunc.parameterInit_imfil_cpu(datadir("img",file), Parameters.parameters_f32);

u0 = reshape(zeros(Float32, p.dim_i, p.dim_j*(5*p.K+2)), p.dim_i, p.dim_j, 5*p.K+2)


f = LaminartFunc.LamFunction_imfil_cpu(
	arr2, #x_lgn,
	arr1, #C,
	similar(arr1), #H_z,
   	similar(arr1), # H_z_temp,
   	similar(arr2), # v_C_temp1,
   	similar(arr2), # v_C_temp2,
   	similar(arr1), # v_C_tempA,
   	similar(arr1[:,:,1]), #W_temp
)
prob = ODEProblem(f, u0, tspan, p)
bm = @benchmark solve(prob)
push!(benches, bm)



# GPU imfilter FIR

p = LaminartInitFunc.parameterInit_imfil_cpu(datadir("img",file), Parameters.parameters_f32);

u0 = reshape(zeros(Float32, p.dim_i, p.dim_j*(5*p.K+2)), p.dim_i, p.dim_j, 5*p.K+2)



f = LaminartFunc.LamFunction_imfil_cpu(
	arr2, #x_lgn,
	arr1, #C,
	similar(arr1), #H_z,
   	similar(arr1), # H_z_temp,
   	similar(arr2), # v_C_temp1,
   	similar(arr2), # v_C_temp2,
   	similar(arr1), # v_C_tempA,
   	similar(arr1[:,:,1]), #W_temp
)
prob = ODEProblem(f, u0, tspan, p)
bm = @benchmark solve(prob)
push!(benches, bm)


# benchmark plot

fig, ax = plt.subplots()
for ben in enumerate(test_name_plt)
    ax.scatter(
        median(benches[ben[1]].times) * 1e-9,
        ben[2],
        color = Utils.colours[ben[1]],
        alpha = 0.3,
        edgecolors = "none",
    )
end


ax.legend()
ax.set_ylabel("Time (\$s\$)")
ax.grid(True)
fig.tight_layout()
plt.savefig(plotsdir(
    string("bench_imp", batch_),
    string("bench_imp_time.png"),
))
close("all")




# memory

fig, ax = plt.subplots()
for ben in enumerate(test_name_plt)
    ax.scatter(
        benches[ben[1]].memory * 1e-6,
        ben[2],
        color = Utils.colours[ben[1]],
        alpha = 0.3,
        edgecolors = "none",
    )
end


ax.legend()
ax.set_ylabel("Memory (\$MB\$)")
ax.grid(True)
fig.tight_layout()
plt.savefig(plotsdir(
    string("bench_imp", batch_),
    string("bench_imp_mem.png"),
))
close("all")


# alloc

fig, ax = plt.subplots()
for ben in enumerate(test_name_plt)
    ax.scatter(
        benches[ben[1]].allocs * 1e-6,
        ben[2],
        color = Utils.colours[ben[1]],
        alpha = 0.3,
        edgecolors = "none",
    )
end


ax.legend()
ax.set_ylabel("Allocations")
ax.grid(True)
fig.tight_layout()
plt.savefig(plotsdir(
    string("bench_imp", batch_),
    string("bench_imp_alloc.png"),
))
close("all")
