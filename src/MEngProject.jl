module MEngProject
include("./Laminart.jl")
include("./LaminartGPU.jl")
include("./LamKernels.jl")
include("./Utils.jl")
include("./Parameters.jl")
using ModelingToolkit, OrdinaryDiffEq
export Laminart, LaminartGPU, LamKernels, Utils, Parameters, testa, kern_A


end
