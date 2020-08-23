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



include("test_illusions.jl")
include("test_noise.jl")
# include("test_parameterVar.jl")
# include("test_kernelsVar.jl")
# include("test_eqAtEquil.jl")
#
# include("bench_solver.jl")
# include("bench_imp.jl")
# include("bench_eq.jl")
# include("bench_dim.jl")
# include("bench_kernels_dim.jl")
