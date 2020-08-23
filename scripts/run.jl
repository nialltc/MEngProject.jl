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



try include("test_illusions.jl") catch err end
print(err)
try include("test_noise.jl") catch err end
print(err)
# try include("test_parameterVar.jl")catch err end
# print(err)
# try include("test_kernelsVar.jl")catch err end
# print(err)
# try include("test_eqAtEquil.jl")catch err end
# print(err)
#
# try include("bench_solver.jl")catch err end
# print(err)
# try include("bench_imp.jl")catch err end
# print(err)
# try include("bench_eq.jl")catch err end
# print(err)
# try include("bench_dim.jl")catch err end
# print(err)
# try include("bench_kernels_dim.jl")catch err end
# print(err)
