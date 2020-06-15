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
using NNlib, ImageFiltering, Images
# , MEngProject.LamKernels

export I_u, fun_v_C, fun_equ

# make resuable static kernels

# function kernels(parameters::Dict)
#     Kernel.gaussian(σ_1, l)
#     return para_dict
#     end

# retina

function I_u(I::AbstractArray, σ_1=1)
    # todo change to DoG for speed and because it doesnt work
    return I - imfilter(I, Kernel.gaussian(σ_1))
end


# todo use saved static(?) of [u+] and [u-]???

# lgn feedback
# todo lgn_A
function fun_lgn_A(x::AbstractArray, kern_sumk, C_1)
    return C_1 .* imfilter(x, kern_sumk(K))
end

# todo
function fun_lgn_B(x::AbstractArray, σ_1, C_2)
    return C_1 .* imfilter(x, kern_sumk(K))
end


# todo check
function fun_f(x::AbstractArray, μ, ν, n)
    (μ .* x .^n) ./ (ν^n .+ x.^n)
end



# LGN
function fun_v(v::AbstractArray, u::AbstractArray, lgn_A::AbstractArray, lgn_B::AbstractArray, δ_v)
    return δ_v .* ( -v +
            ((1 - v) * relu(u) * (1 + lgn_A)) -
            ((1 + v) * lgn_B))
end


# lgn to l6 and l4
function fun_v_C(v_p::AbstractArray, v_m::AbstractArray, σ::Real, K::Int, γ=10, l = 4*ceil(Int,σ)+1)
    # isodd(l) || throw(ArgumentError("length must be odd"))

# todo replace gaussian kernel with static
    V = exp(-1/8) .* (imfilter((relu.(v_p)-relu.(v_m)), Kernel.gaussian(σ), "circular"))

    A = reshape(Array{eltype(V)}(undef, size(V)[1], size(V)[2]*K),size(V)[1],size(V)[2],K)
    B = copy(A)

# todo replace kern_A() and kern_B with premade kernels here
    for k in 1:K
        θ = π*(k-1)/K
        A[:,:,k] = imfilter(V, LamKernels.kern_A(σ, θ), "circular")
        B[:,:,k] = abs.(imfilter(V, LamKernels.kern_B(σ, θ), "circular"))
    end

    return γ .* (relu.(A .- B) .+ relu.(.- A .- B))
end



# L6
function fun_dx_V1(x::AbstractArray, C::AbstractArray, z::AbstractArray, x_v2::AbstractArray, α, ϕ, Γ, δ_c, V_21, att=0)
    return δ_c .* (-x .+
            ((1 .- x) .*
                ((α*C) .+ (ϕ .* relu.(z .- Γ)) .+ (V_21 .* x_v2) .+ att)))
end



#     L4 excit
function fun_dy(y::AbstractArray, C::AbstractArray, x::AbstractArray, m::AbstractArray, η_p, kern_W_p, μ, ν, n, δ_c)
    return  δ_c(   -y .+
            ((1 .- y) .* (C .+ (η_p .* x))) .-
            ((1 .+ y) .* fun_f((m .* imfilter(m, kern_W_p), μ, ν, n))))
end



#     l4 inhib
function fun_dm(m::AbstractArray, x::AbstractArray, kern_W_m, η_m, μ, ν, n, δ_m)
    return δ_m .* (  -m .+
                     (η_m .* x) -
                     (m .* fun_f.(imfilter(m, kern_W_m), μ, ν, n)))
end



#     L2/3 excit
function fun_dz(z::AbstractArray, y::AbstractArray, T_p, H, λ, Γ, ψ, δ_z, a_ex_23, att)
    return δ_z .*   (-z .+
                    ((1 .- z) .*
                        ((λ .* relu.(y)) .+ imfilter((relu.(z .- Γ)), H) .+ (a_ex_23 .* att))) .-
                    ((z .+ ψ) .* (imfilter(s, T_p))))
end



#     L2/3 inhib
function fun_ds(s::AbstractArray, z::AbstractArray, H, a_23_in, T_m, δ_s, att)
    return δ_s .*   ( -s .+
                    imfilter((relu.(z .- Γ)), H) .+ (a_23_in .* att) .-
                    (s .* imfilter(s, T_m)))  #?????
end



#     V2 L6
function fun_dx_v2(x_v2::AbstractArray, z_v2::AbstractArray, z::AbstractArray, v_12_6, Γ, ϕ, δ_c, att)
    return δ_c .*   (  -x_v2 .+
                    ((1 .- x_v2) .*
                        ((v_12_6 .* relu.(z .- Γ)) + (ϕ .* relu.(z_v2 .- Γ)) .+ att)))
end


# V2 L4 excit
function fun_dy_v2(y_v2::AbstractArray, z::AbstractArray, x_v2::AbstractArray, m_v2::AbstractArray, W_p, v_12_4, Γ, η_p, μ, ν, n, δ_c)
    return δ_c .*   (-y_v2 .+
                    ((1 .- y_v2) .* ((v_12_4 .* relu.(z .- Γ)) .+ (η_p .* x_v2))) .-
                    ((1 .+ y_v2) .* fun_f.(imfilter(m_v2, W_p), μ, ν, n)))
end




# for equilabrium
fun_equ(x) = x/(1+x)


# lgn
function fun_v_equ(u::AbstractArray, x::AbstractArray, lgn_para_u=1, lgn_para_A=0, lgn_para_B=0)
    return fun_equ.(relu.(u) .* (lgn_para_u .+ (lgn_para_A .* fun_A.(x))) .- (lgn_para_B .* fun_B.(x)))
end


# lgn, no L6 feedback, light
function fun_v_equ_noFb(u::AbstractArray, x::AbstractArray, lgn_para_u=1)
    return fun_equ.(relu.(u) .* (lgn_para_u))
end


# l6
function fun_x_equ(C::AbstractArray, z::AbstractArray,  x_v2::AbstractArray, ϕ, Γ, V_21=0, att=0)
    return fun_equ.((α .* C) .+ (ϕ .* max.(z,Γ)) + (V_21 .* x_v2) .+ att)
end


# l6, no V2 feedback, light
function fun_x_equ_noV2(C::AbstractArray, z::AbstractArray, ϕ, Γ)
    return fun_equ.((α .* C) .+ (ϕ .* max.(z,Γ)))
end


# l4 excit
function fun_y_equ(C::AbstractArray, x::AbstractArray,  m::AbstractArray, W_p, η_p)
    return fun_equ.(C .+ (η_p .* x - fun_f.(imfilter(m, W_p), μ, ν, n)))
end

# l4 inhib, needs initial condition of itself
function fun_m_equ(C::AbstractArray, x::AbstractArray, m_init::AbstractArray, W_p, η_p, μ, ν, n)
    return (η_m .* x ./ (1 .+  fun_f.(imfilter(m_init, W_p), μ, ν, n)))
end


# l4 inhib - no feedback kernel
function fun_m_equ_noFb(C::AbstractArray, x::AbstractArray, η_p)
    return (η_m .* x)
end


#  l2/3 excit, needs initial condition of itself
function fun_z_equ(y::AbstractArray, s::AbstractArray,  z_init::AbstractArray, γ, H, T_p, Γ, ϕ, att=0, a_23_ex=3)
    return (λ .* relu.(y)) .+ imfilter(max.(z_init, Γ), H) .+ (a_23_ex .* att) .- (ϕ .* imfilter(s, T_p)) ./
    (1 .+ (λ .* relu.(y)) .+ imfilter(max.(z_init, Γ), H) .+ (a_23_ex .* att) .+ imfilter(s, T_p))
    return
end


#  l2/3 excit - no feedback kernel
function fun_z_equ_noFb(y::AbstractArray, s::AbstractArray, γ, H, T_p, Γ, ϕ)
    return (λ .* relu.(y)) .- (ϕ .* imfilter(s, T_p)) ./
    (1 .+ (λ .* relu.(y)) .+ imfilter(s, T_p))
    return
end


#  l2/3 inhib, needs initial condition of itself
function fun_s_equ(z::AbstractArray, s::AbstractArray,  s_init::AbstractArray, Γ, H, T_m, att=0, a_23_in)
    return (imfilter(max.(z, Γ), H) + a_23_in .* att ) ./ (1 .+ imfilter(s_init, T_m))            #????????? is T right?
end


#  l2/3 inhib - no feedback kernel
function fun_s_equ_noFb(z::AbstractArray, s::AbstractArray, Γ, H, T_m, att=0, a_23_in)
    return (imfilter(max.(z, Γ), H))
end


# V2 L6
function fun_xV2_equ(x_v2::AbstractArray, z::AbstractArray, z_v2::AbstractArray, ϕ, Γ, V_12_6, att=0)
    return fun_equ.((V_12_6 .* max.(z, Γ) .+ (ϕ .* max.(z_v2, Γ))))
end


#  V2 L4 excit
function fun_yV2_equ(z::AbstractArray, x::AbstractArray,  m::AbstractArray, W_p, η_p, Γ, μ, ν, n, V_12_4, att=0)
    return (V_12_4 .* max.(z,Γ) .+ (η_p .* x) - fun_f.(imfilter(m, W_p), μ, ν, n) ./
   (1 .+ V_12_4 .* max.(z,Γ) .+ (η_p .* x) .+ fun_f.(imfilter(m, W_p), μ, ν, n)
end

end

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


