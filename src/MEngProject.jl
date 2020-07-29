module MEngProject
include("./Laminart.jl")
include("./LaminartGPU.jl")
include("./LaminartConv.jl")
include("./Laminartv1.jl")
include("./LamKernels.jl")
include("./Utils.jl")
include("./Parameters.jl")
using ModelingToolkit, OrdinaryDiffEq, CUDA
export Laminart, LaminartGPU, LaminartConv, LamKernels, Laminartv1, Utils, Parameters, testa, kern_A


end
