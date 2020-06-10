"""
# module laminart

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-06-07

# Examples

```jldoctest
julia>
```
"""
module Laminart
    include("./LamKernels.jl")
using NNlib, ImageFiltering, Images, MEngProject.LamKernels

export I_u, fun_v_C, fun_equ
# retina

function I_u(I::AbstractArray, σ_1=1)
    # todo change to DoG for speed and because it doesnt work
    I - imfilter(I, Kernel.gaussian(σ_1))
end


# LGN

# dv_p = δ_v( -v_p +
#             ((1 - v_p) * relu(u) * (1 + lgn_A)) -
#             ((1 + v_p) * lgn_B))
#
#
# dv_m = δ_v( -v_m +
#             ((1 - v_m) * relu(-u) * (1 + lgn_A)) -
#             ((1 + v_m) * lgn_B))
#
# # feedback to lgn
# # todo lgn_A
# lgn_A = C_1 * imfilter(x, kern_sumk(K))
#
# lgn_B = C_2 * imfilter(x, Kernel.gaussian(σ_1), "circular")


# lgn to l6 and l4

function fun_v_C(v_p::AbstractArray, v_m::AbstractArray, σ::Real, K::Int, γ=10, l = 4*ceil(Int,σ)+1)
    # isodd(l) || throw(ArgumentError("length must be odd"))

    V = exp(-1/8) .* (imfilter((relu.(v_p)-relu.(v_m)), Kernel.gaussian(σ), "circular"))

    A = reshape(Array{eltype(V)}(undef, size(V)[1], size(V)[2]*K),size(V)[1],size(V)[2],K)
    B = copy(A)

# todo replace kern_A() and kern_B with premade kernels here
    for k in 1:K
        θ = π*(k-1)/K
        A[:,:,k] = imfilter(V, MEngProject.LamKernels.kern_A(σ, θ), "circular")
        B[:,:,k] = abs.(imfilter(V, MEngProject.LamKernels.kern_B(σ, θ), "circular"))
    end

    γ .* (relu.(A .- B) .+ relu.(.- A .- B))
end

#
# # L6
#
# dx = δ_c(   -x +
#             ((1 -x) *
#                 ((α*C) + (ϕ * relu.(z - Γ)) + (V_21 * x_v2) + att)))
#
# # L4
#
# dy = δ_c(   -y +
#             ((1 - y) * (C + (η_p * x))) -
#             ((1 + y) * fun_f((m .* imfilter(m, kern_W_p)))))
#
#
# # williamson - eq 20, relu? unclear
# # dy = δ_c(   -y +
# #             ((1 - y) * relu.((C + (η_p * x)))) -
# #             (y + 1) * sigmoid((m .* imfilter(m, W_p))))
#
#
# dm = δ_m(   -m +
#             (η_m * x) -
#             (m .* fun_f(imfilter(m, kern_W_m))))
#
#
# # L2/3
#
# dz = δ_z(   -z +
#             ((1 - z) *
#                 ((λ * relu(y)) + imfilter((relu(z - Γ)), H) + (a_ex_23 * att))) -
#             ((z + ψ) .* (imfilter(s, T_p))))
#
#
# # todo sTs
# ds = δ_s(   -s +
#             imfilter((relu. (z - Γ)), H) + (a_23_in .* att) -
#             (s .* imfilter(s, T_m)))  #?????
#
#
#
#
# # V1 to V2
#
# # V2 L6
#
# dx_v2 = δ_c(    -x_v2 +
#                 (1 - x_v2) *
#                     ((v_12_6 * relu.(z - Γ)) + (ϕ * relu.(z_v2 - Γ)) + att))
#
#
# # V2 L4 excit
#
# dy_v2 = δ_c(    -y_v2 +
#                 ((1 - y_v2) * ((v_12_4 * relu.(z - Γ)) + (η_p * x_v2))) -
#                 ((1 + y_v2) * fun_f(imfilter(m_v2, W_p))))
#
#
# # V2 L4 inhib
#
#
# dm_v2 = δ_m(    -m_v2 +
#                 (η_m * x_v2) -
#                 (m_v2 .* fun_f(imfilter(m_v2, W_m))))
#
#
# # L2/3
#
# dz_v2 = δ_z(    -z_v2 +
#                 ((1 - z_v2) *
#                     ((λ * relu(y_v2)) +
#                         imfilter((relu.(z_v2 - Γ)), H) + a_ex_23 * att)) -
#                 ((z_v2 + ψ) .* (imfilter(s_v2, T_p))))
#
#
# # todo sTs
# ds_v2 = δ_s(    -s_v2 +
#                 imfilter((max(z_v2,Γ)),H_v2) -
#                 (s_v2 .* imfilter(s_v2, T_m)))  #?????

# # todo
# function fun_f(x::AbstractArray, μ::Real = Global μ,
#     ν::Real = Global ν, n::Real = Global n)
#     μ .* x .^n ./(ν^n .+ x.^n)
# end


# for equilabrium
fun_equ(x) = x/(1+x)

end