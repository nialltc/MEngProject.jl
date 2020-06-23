module MEngProject
include("./Laminart.jl")
include("./LamKernels.jl")
include("./Utils.jl")
include("./Parameters.jl")
using ModelingToolkit, OrdinaryDiffEq
export Laminart, LamKernels, Utils, Parameters, testa, kern_A


end
