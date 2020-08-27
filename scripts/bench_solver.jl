"""
# script bench_solver

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-08-20


Script to benchmark GPU and CPU implementions of model.
# Examples

```jldoctest
julia>
```
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
    ImageIO,
    MEngProject.LaminartKernels,
    MEngProject.LaminartInitFunc,
    MEngProject.Utils,
    BenchmarkTools,
    Test

using OrdinaryDiffEq,
    ParameterizedFunctions, LSODA, Sundials, DiffEqDevTools, Noise

batch = 9000


# global benchm_s = []
# global benchm_sname = []
# global alg_ = []
# global stats_ = []
# global gpu_ = []
# let

benchm_s = []
alg_ = []
stats_ = []
gpu_ = []


tspan = (0.0f0, 800f0)

batch_ = string(batch, "_", rand(1000:9999))
mkdir(plotsdir(string("bench_solver", batch_)))
file = "kan_sq_cont_l.png"

solvers = [
    #             AutoTsit5(Rosenbrock23()),
    Tsit5(),
    BS3(),
    lsoda(),
    #             Vern7(),
]
# alg=lsoda()

# GPU
p = LaminartInitFunc.parameterInit_conv_gpu(
    datadir("img", file),
    Parameters.parameters_f32,
);

u0 = cu(reshape(
    zeros(Float32, p.dim_i, p.dim_j * (5 * p.K + 2)),
    p.dim_i,
    p.dim_j,
    5 * p.K + 2,
    1,
))

arr1 = similar(@view u0[:, :, 1:2, :])
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
global prob_s = ODEProblem(f, u0, tspan, p)

# alg = solvers[1]
for algg in solvers
    @. u0 = 0.0f0
    try
        global algg_ = algg
        sol = solve(prob_s, algg)
        push!(alg_, sol.alg)
        push!(stats_, sol.destats)
        print(sol.destats)
        push!(benchm_s, @benchmark solve(prob_s, algg_))
        push!(gpu_, "GPU")
    catch err
        print(err)
    end
end






# CPU conv

p = LaminartInitFunc.parameterInit_imfil_cpu(
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


f = LaminartFunc.LamFunction_imfil_cpu(
    similar(arr1), # H_z_temp,
    similar(arr2[:, :, 1]), # v_C_temp1,
    similar(arr2[:, :, 1]), # v_C_temp2,
    similar(arr1), # v_C_tempA,
    similar(arr1[:, :, 1]), #W_temp
)
global prob_sc = ODEProblem(f, u0, tspan, p)
for algg in solvers
    @. u0 = 0.0f0
    try
        global algg_ = algg
        sol = solve(prob_sc, algg)
        push!(alg_, sol.alg)
        push!(stats_, sol.destats)
        print(sol.destats)
        push!(benchm_s, @benchmark solve(prob_s, algg_))
        push!(gpu_, "CPU")
    catch err
        print(err)
    end
end



# benchmark plot

# time
fig, ax = plt.subplots()
for ben in enumerate(gpu_)
    ax.scatter(
        string(ben[2], " ", alg_[ben[1]]),
        median(benchm_s[ben[1]].times) * 1e-9,
        color = Utils.colours[ben[1]],
        edgecolors = "none",
    )
end

ax.set_ylabel("Time (\$s\$)")
ax.set_ylim(ymin = 0)
ax.grid(true)
fig.tight_layout()
plt.savefig(plotsdir(
    string("bench_solver", batch_),
    string("bench_solver_time.png"),
))
close("all")




# memory

fig, ax = plt.subplots()
for ben in enumerate(gpu_)
    ax.scatter(
        string(ben[2], " ", alg_[ben[1]]),
        benchm_s[ben[1]].memory * 1e-6,
        color = Utils.colours[ben[1]],
        edgecolors = "none",
    )
end

ax.set_ylabel("Memory (\$MB\$)")
ax.set_ylim(ymin = 0)
ax.grid(true)
fig.tight_layout()
plt.savefig(plotsdir(
    string("bench_solver", batch_),
    string("bench_solver_mem.png"),
))
close("all")


# alloc

fig, ax = plt.subplots()
for ben in enumerate(gpu_)
    ax.scatter(
        string(ben[2], " ", alg_[ben[1]]),
        benchm_s[ben[1]].allocs,
        color = Utils.colours[ben[1]],
        edgecolors = "none",
    )
end

ax.set_ylabel("Allocations")
ax.set_ylim(ymin = 0)
ax.grid(true)
fig.tight_layout()
plt.savefig(plotsdir(
    string("bench_solver", batch_),
    string("bench_solver_alloc.png"),
))
close("all")


# end
