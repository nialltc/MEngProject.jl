# """
# # script run
#
# - Julia version: 1.4
# - Author: niallcullinane
# - Date: 2020-08-20
#
#
# Runs set of tests and benchmarks.
# Commits and pushes results to Git.
# # Examples
#
# ```jldoctest
# julia>
# ```
# """
#
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

# CUDA.allowscalar(false)

try include("test_illusions.jl") catch err; print(err) end
CUDA.reclaim()
try include("test_noise.jl") catch err; print(err) end
CUDA.reclaim()
try include("test_parameterVar.jl") catch err; print(err) end
CUDA.reclaim()
try include("test_kernelsVar.jl") catch err; print(err) end
CUDA.reclaim()
# try include("test_eqAtEquil.jl")catch err; print(err) end
CUDA.reclaim()
#
#
# # try include("bench_solver.jl")catch err; print(err) end
try include("bench_imp.jl") catch err; print(err) end
CUDA.reclaim()
# try include("bench_eq.jl")catch err; print(err) end
try include("bench_kernels_dim.jl") catch err; print(err) end
CUDA.reclaim()
try include("bench_dim.jl") catch err; print(err) end
CUDA.reclaim()
#
# run(`git pull`)
# run(`git add plots/*`)
# run(`git commit -m "test"`)
# run(`git push`)
