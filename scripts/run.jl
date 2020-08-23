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



try include("test_illusions.jl") catch err; print(err) end
CUDA.reclaim()
try include("test_noise.jl") catch err; print(err) end
CUDA.reclaim()
try include("test_parameterVar.jl")catch err; print(err) end
CUDA.reclaim()
try include("test_kernelsVar.jl")catch err; print(err) end
CUDA.reclaim()
# try include("test_eqAtEquil.jl")catch err; print(err) end
CUDA.reclaim()


# try include("bench_solver.jl")catch err; print(err) end
try include("bench_imp.jl")catch err; print(err) end
CUDA.reclaim()
# try include("bench_eq.jl")catch err; print(err) end
try include("bench_kernels_dim.jl")catch err; print(err) end
CUDA.reclaim()
try include("bench_dim.jl")catch err; print(err) end
CUDA.reclaim()