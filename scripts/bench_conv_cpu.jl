"""
# script bench_dim

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-08-20


Script to benchmark model with different input sizes for CPU and GPU.
"""

using DrWatson
@quickactivate "MEngProject"
using MEngProject,
    CUDA,
    DifferentialEquations,
    PyPlot,
    NNlib,
    ImageFiltering,
    Images,
    MEngProject.LaminartKernels,
    MEngProject.LaminartInitFunc,
    MEngProject.Utils,
    BenchmarkTools,
    Test

using OrdinaryDiffEq,
    ParameterizedFunctions, LSODA, Sundials, DiffEqDevTools, Noise

benchm_co = []
tspan = (0.0f0, 10f0)

batch_ = string(batch, "_", rand(1000:9999))
mkdir(plotsdir(string("bench_conv", batch_)))
file = "kan_sq_cont_l.png"

test_name_plt = [
    "CPU conv",
    "CPU imfilter",
    "GPU imfilter FFT",
    "GPU imfilter IIR",
    "GPU imfilter FIR",
]





# CPU conv

global p = LaminartInitFunc.parameterInit_conv_cpu(
    datadir("img", file),
    Parameters.parameters_f32,
);

u0 = reshape(
    zeros(Float32, p.dim_i, p.dim_j * (5 * p.K + 2)),
    p.dim_i,
    p.dim_j,
    5 * p.K + 2,
    1,
)

arr1 = similar(@view u0[:, :, 1:p.K, :])
arr2 = similar(@view u0[:, :, 1:1, :])

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
global prob_ke = ODEProblem(f, u0, tspan, p)
#         push!(benchm_ke, @benchmark solve(prob_ke))
global du = similar(u0)
global u_ = u0
push!(benchm_co, @benchmark f(du, u_, p, 1.0f0))

# CPU imfilter

global p = LaminartInitFunc.parameterInit_imfil_cpu(
    datadir("img", file),
    Parameters.parameters_f32,
);

u0 = reshape(
    zeros(Float32, p.dim_i, p.dim_j * (5 * p.K + 2)),
    p.dim_i,
    p.dim_j,
    5 * p.K + 2,
);


arr1 = u0[:, :, 1:p.K]
arr2 = u0[:, :, 1:1];


f = LaminartFunc.LamFunction_imfil_cpu(
    similar(arr2[:, :, 1]), #x_lgn,
    arr1, #C,
    similar(arr1), #H_z,
    similar(arr1), # H_z_temp,
    similar(arr2[:, :, 1]), # v_C_temp1,
    similar(arr2[:, :, 1]), # v_C_temp2,
    similar(arr1), # v_C_tempA,
    similar(arr1[:, :, 1]), #W_temp
);

global prob_ke = ODEProblem(f, u0, tspan, p)
#         push!(benchm_ke, @benchmark solve(prob_ke))
global du = similar(u0)
global u_ = u0
push!(benchm_co, @benchmark f(du, u_, p, 1.0f0))

# GPU imfilter FFT

global p = LaminartInitFunc.parameterInit_imfil_cpu(
    datadir("img", file),
    Parameters.parameters_f32,
);

u0 = reshape(
    zeros(Float32, p.dim_i, p.dim_j * (5 * p.K + 2)),
    p.dim_i,
    p.dim_j,
    5 * p.K + 2,
);


arr1 = u0[:, :, 1:p.K]
arr2 = u0[:, :, 1:1];


f = LaminartFunc.LamFunction_imfil_gpu_fft(
    similar(arr2[:, :, 1]), #x_lgn,
    arr1, #C,
    similar(arr1), #H_z,
    similar(arr1), # H_z_temp,
    similar(arr2[:, :, 1]), # v_C_temp1,
    similar(arr2[:, :, 1]), # v_C_temp2,
    similar(arr1), # v_C_tempA,
    similar(arr1[:, :, 1]), #W_temp
);

global prob_ke = ODEProblem(f, u0, tspan, p)
#         push!(benchm_ke, @benchmark solve(prob_ke))
global du = similar(u0)
global u_ = u0
push!(benchm_co, @benchmark f(du, u_, p, 1.0f0))

# GPU imfilter IIR

global p = LaminartInitFunc.parameterInit_imfil_cpu(
    datadir("img", file),
    Parameters.parameters_f32,
);

u0 = reshape(
    zeros(Float32, p.dim_i, p.dim_j * (5 * p.K + 2)),
    p.dim_i,
    p.dim_j,
    5 * p.K + 2,
);


arr1 = u0[:, :, 1:p.K]
arr2 = u0[:, :, 1:1];


f = LaminartFunc.LamFunction_imfil_gpu_iir(
    similar(arr2[:, :, 1]), #x_lgn,
    arr1, #C,
    similar(arr1), #H_z,
    similar(arr1), # H_z_temp,
    similar(arr2[:, :, 1]), # v_C_temp1,
    similar(arr2[:, :, 1]), # v_C_temp2,
    similar(arr1), # v_C_tempA,
    similar(arr1[:, :, 1]), #W_temp
);

global prob_ke = ODEProblem(f, u0, tspan, p)
#         push!(benchm_ke, @benchmark solve(prob_ke))
global du = similar(u0)
global u_ = u0
push!(benchm_co, @benchmark f(du, u_, p, 1.0f0))

# GPU imfilter FIR

global p = LaminartInitFunc.parameterInit_imfil_cpu(
    datadir("img", file),
    Parameters.parameters_f32,
);

u0 = reshape(
    zeros(Float32, p.dim_i, p.dim_j * (5 * p.K + 2)),
    p.dim_i,
    p.dim_j,
    5 * p.K + 2,
);


arr1 = u0[:, :, 1:p.K]
arr2 = u0[:, :, 1:1];


f = LaminartFunc.LamFunction_imfil_gpu_fir(
    similar(arr2[:, :, 1]), #x_lgn,
    arr1, #C,
    similar(arr1), #H_z,
    similar(arr1), # H_z_temp,
    similar(arr2[:, :, 1]), # v_C_temp1,
    similar(arr2[:, :, 1]), # v_C_temp2,
    similar(arr1), # v_C_tempA,
    similar(arr1[:, :, 1]), #W_temp
);

global prob_ke = ODEProblem(f, u0, tspan, p)
#         push!(benchm_ke, @benchmark solve(prob_ke))
global du = similar(u0)
global u_ = u0
push!(benchm_co, @benchmark f(du, u_, p, 1.0f0))

# benchmark plot

# time
fig, ax = plt.subplots()
for ben in enumerate(test_name_plt)
    ax.scatter(
        ben[2],
        median(benchm_co[ben[1]].times) * 1e-9,
        color = Utils.colours[ben[1]],
        edgecolors = "none",
    )
end

ax.set_ylabel("Time (\$s\$)")
ax.set_ylim(ymin = 0)
ax.grid(true)
fig.tight_layout()
plt.savefig(plotsdir(
    string("bench_conv", batch_),
    string("bench_conv_time.png"),
))
close("all")




# # memory

# fig, ax = plt.subplots()
# for ben in enumerate(test_name_plt)
#     ax.scatter(
#         ben[2],
#         benchm_i[ben[1]].memory * 1e-6,
#         color = Utils.colours[ben[1]],
#         edgecolors = "none",
#     )
# end

# ax.set_ylabel("Memory (\$MB\$)")
# ax.set_ylim(ymin=0)
# ax.grid(true)
# fig.tight_layout()
# plt.savefig(plotsdir(string("bench_conv", batch_), string("bench_convmem.png")))
# close("all")


# # alloc

# fig, ax = plt.subplots()
# for ben in enumerate(test_name_plt)
#     ax.scatter(
#         ben[2],
#         benchm_i[ben[1]].allocs,
#         color = Utils.colours[ben[1]],
#         edgecolors = "none",
#     )
# end

# ax.set_ylabel("Allocations")
# ax.set_ylim(ymin=0)
# ax.grid(true)
# fig.tight_layout()
# plt.savefig(plotsdir(
#     string("bench_conv", batch_),
#     string("bench_conv_alloc.png"),
# ))
# close("all")
benchm_gpu = nothing
benchm_cpu = nothing
y1Res_gpu = nothing
y1Res_cpu = nothing
prob_d = nothing
CUDA.reclaim()
