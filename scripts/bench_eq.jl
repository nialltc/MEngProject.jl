"""
# script bench_eq

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-08-20


Script to microbenchmark model equations.
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

batch = 1


files = readdir(datadir("img"))


tspan = (0.0f0, 800f0)

batch_ = string(batch, "_", rand(1000:9999))
mkdir(plotsdir(string("bench_eq", batch_)))
file = "kan_sq_cont_l.png"

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

arr1 = similar(u0[:, :, 1:2, :])
arr2 = similar(u0[:, :, 1:1, :])

f = LaminartGPU.LamFunction_equ(
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
);
prob = ODEProblem(f, u0, tspan, p)
bm = @benchmark solve(prob)
push!(benches, bm)
