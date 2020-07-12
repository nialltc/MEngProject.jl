using MEngProject
using Test
using Images
img = convert(Array{Float32,2}, load("../input_img/Iine_100_100_gs.png"))
p = Laminart.kernels(img, Parameters.parameters)
p = Laminart.add_I_u_p(img, p)

@testset "MEngProject.jl" begin

out = zeros(p.dim_i, p.dim_j)
b = reshape(ones(p.dim_i, p.dim_j*p.K), p.dim_i, p.dim_j, p.K)
c = 2 .* ones(p.dim_i, p.dim_j)


@test Laminart.fun_x_lgn!(out, b, p) ≈ c
# @test Laminart.fun_dv!(out,)
# @test Laminart.fun_v_C!(out)
# @test Laminart.fun_
# @test Laminart.fun_
# @test Laminart.fun_
# @test Laminart.fun_
# @test Laminart.fun_
# @test Laminart.fun_
# @test Laminart.fun_
# @test Laminart.fun_
# @test Laminart.fun_
# @test Laminart.fun_
# @test Laminart.fun_
# @test Laminart.fun_


end