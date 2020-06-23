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

# todo: make resuable static kernels

function kernels(p::NamedTuple)
    C_A_temp = reshape(Array{Real}(undef, p.C_AB_l, p.C_AB_l * p.K),p.C_AB_l,p.C_AB_l,p.K)
    C_B_temp = copy(C_A_temp)
    H_temp = reshape(Array{Real}(undef, p.H_l, p.H_l * p.K),p.H_l,p.H_l,p.K)
    for k ∈ 1:p.K
        θ = π*(k-1)/p.K
        C_A_temp[:,:,k] = LamKernels.kern_A(p.σ_2, θ)               #ij ijk ijk
        C_B_temp[:,:,k] = LamKernels.kern_B(p.σ_2, θ)               #ij ijk ijk
        H_temp[:,:,k] = reflect(p.H_fact .* LamKernels.gaussian_rot(p.H_σ_x, p.H_σ_y, θ, p.H_l))  #ijk, ij for each k; ijk
#         W_temp[:,:,:,k] =
    end
    temp_out = (
    gauss_1 = reflect(Kernel.gaussian(p.σ_1)),
    gauss_2 = reflect(Kernel.gaussian(p.σ_2)),
    C_A = reflect(C_A_temp),
    C_B = reflect(C_B_temp),
#     W_p = kern_W_p(),
#     W_m = kern_W(),
    H = H_temp)
#     T_p = reflect(kern_T_p()),
#     T_m = reflect(kern_T_m()))
    return merge(p, temp_out)
    end

# todo union parameters and kernels?

# retina

function I_u(I::AbstractArray, p::Tuple)
    # todo: change to DoG for speed and because it doesnt work
    return I - imfilter(I, p.gauss_1)
end


# todo: use saved static(?) of [u+] and [u-]???

# lgn feedback

# todo: test
# todo: should lgn_a be normalized, ie divide by k??

function fun_x_lgn(x::AbstractArray)
   # todo: change to abstract array? or is eltype doing that??
    x_LGN =Array{eltype(V)}(undef, size(V)[1], size(V)[2])
#     todo: change to map function?
    for k in 1:size(x)[3]
        x_lgn .+= x[:,:,k]
    end
    return x_lgn
end

# # lgn_A
# # todo: test
# function x_lgn_A(x_lgn, C1)
#     return C_1 .* x_A
# end
#
# # lgn_B
# # todo: test
# function fun_lgn_B(x_lgn::AbstractArray, σ_1, C_2)
# #     todo: alocate kernel
#     return C_2 .* imfilter(x_A, Kernel.gaussian(σ_1))
# end



function fun_F(value::Real, p::Tuple)
    max.(value - p.Γ, 0)
end


# williomson uses differnt F, relu with threshold
function fun_F_willimson(value::Real, p::Tuple)
    value < p.Γ ? zero(value) : value
end




# todo: check
function fun_f(x::AbstractArray, p::Tuple)
    (p.μ .* p.x .^p.n) ./ (p.ν^p.n .+ p.x.^p.n)
end



# LGN
function fun_dv(v::AbstractArray, u::AbstractArray, x::AbstractArray, p::Tuple)
    x_lgn = fun_x_lgn(x)
    return p.δ_v .* ( -p.v +
            ((1 - p.v) * relu(u) * (1 + p.C1 * x_lgn)) -
            ((1 + p.v) * p.C2 * imfilter(x_lgn, p.gauss_1, "circular")))
end


# lgn to l6 and l4
function fun_v_C(v_p::AbstractArray, v_m::AbstractArray, p::Tuple)
    # isodd(l) || throw(ArgumentError("length must be odd"))

    V = exp(-1/8) .* (imfilter((relu.(v_p)-relu.(v_m)), p.gauss_2, "circular"))

# todo: change to abstract array? or is eltype doing that??
    A = reshape(Array{eltype(V)}(undef, size(V)[1], size(V)[2]*p.K),size(V)[1],size(V)[2],p.K)
    B = copy(A)

    for k in 1:p.K
        θ = π*(k-1)/p.K
        A[:,:,k] = imfilter(V, p.C_A, "circular")
        B[:,:,k] = abs.(imfilter(V, p.C_B, "circular"))
    end

    return γ .* (relu.(A .- B) .+ relu.(.- A .- B))
end



# L6
function fun_dx_V1(x::AbstractArray, C::AbstractArray, z::AbstractArray, x_v2::AbstractArray, p::Tuple)
    return p.δ_c .* (-x .+
            ((1 .- x) .*
                ((p.α*C) .+ (p.ϕ .* relu.(z .- p.Γ)) .+ (p.V_21 .* x_v2) .+ p.att)))
end



#     L4 excit
function fun_dy(y::AbstractArray, C::AbstractArray, x::AbstractArray, m::AbstractArray, p::Tuple)
    return  p.δ_c(   -y .+
            ((1 .- y) .* (C .+ (p.η_p .* x))) .-
            ((1 .+ y) .* fun_f((m .* imfilter(m, p.W_p), p.μ, p.ν, p.n))))
end



#     l4 inhib
function fun_dm(m::AbstractArray, x::AbstractArray, p::Tuple)
    return p.δ_m .* (  -m .+
                     (p.η_m .* x) -
                     (m .* fun_f.(imfilter(m, p.W_m), p.μ, p.ν, p.n)))
end



#     L2/3 excit
function fun_dz(z::AbstractArray, y::AbstractArray,  H_z::AbstractArray, p::Tuple)
    return p.δ_z .*   (-z .+
                    ((1 .- z) .*
                        ((p.λ .* relu.(y)) .+ H_z .+ (a_ex_23 .* att))) .-
                    ((z .+ ψ) .* (imfilter(s, p.T_p))))
end



#     L2/3 inhib
function fun_ds(s::AbstractArray, z::AbstractArray, H_z::AbstractArray, p::Tuple)
    return p.δ_s .*   ( -s .+
                    H_z .+ (a_23_in .* att) .-
                    (s .* imfilter(s, p.T_m)))  #?????
end

function fun_H_z(z::AbstractArray, p::Tuple)
    H_z_out = copy(z)
    for k ∈ 1:p.K
        H_z_out[:,:,k] = imfilter((relu.(z[:,:,k] .- p.Γ)), reflect(p.kern_H[:,:,k]))
        end
        return H_z_out
end



#     V2 L6
function fun_dx_v2(x_v2::AbstractArray, z_v2::AbstractArray, z::AbstractArray, p::Tuple)
    return p.δ_c .*   (  -x_v2 .+
                    ((1 .- x_v2) .*
                        ((v_12_6 .* relu.(z .- p.Γ)) + (p.ϕ .* relu.(z_v2 .- p.Γ)) .+ p.att)))
end


# V2 L4 excit
function fun_dy_v2(y_v2::AbstractArray, z::AbstractArray, x_v2::AbstractArray, m_v2::AbstractArray, p::Tuple)
    return δ_c .*   (-y_v2 .+
                    ((1 .- y_v2) .* ((v_12_4 .* relu.(z .- p.Γ)) .+ (p.η_p .* x_v2))) .-
                    ((1 .+ y_v2) .* fun_f.(imfilter(m_v2, p.W_p)), p.μ, p.ν, p.n)))
end




# for equilabrium
fun_equ(x) = x/(1+x)


# lgn
function fun_v_equ(u::AbstractArray, x::AbstractArray, p::Tuple)
    return fun_equ.(relu.(u) .* (p.lgn_equ_u .+ (p.lgn_equ_A .* fun_A.(x))) .- (p.lgn_equ_B .* fun_B.(x)))
end


# lgn, no L6 feedback, light
function fun_v_equ_noFb(u::AbstractArray, x::AbstractArray, lgn_para_u=1)
    return fun_equ.(relu.(u) .* (lgn_para_u))
end


# l6
function fun_x_equ(C::AbstractArray, z::AbstractArray,  p::Tuple)
    return fun_equ.((α .* C) .+ (ϕ .* fun_F.(z,Γ)) + (V_21 .* x_v2) .+ att)
end


# l6, no V2 feedback, light
function fun_x_equ_noV2(C::AbstractArray, z::AbstractArray, p::Tuple)
    return fun_equ.((α .* C) .+ (ϕ .* fun_F.(z,Γ)))
end


# l4 excit
function fun_y_equ(C::AbstractArray, x::AbstractArray,  m::AbstractArray, p::Tuple)
    return fun_equ.(C .+ (η_p .* x - fun_f.(imfilter(m, reflect(kern_W_p)), μ, ν, n)))
end

# l4 inhib, needs initial condition of itself
function fun_m_equ(C::AbstractArray, x::AbstractArray, m_init::AbstractArray, p::Tuple)
    return (η_m .* x ./ (1 .+  fun_f.(imfilter(m_init, reflect(kern_W_m)), μ, ν, n)))
end


# l4 inhib - no feedback kernel
function fun_m_equ_noFb(C::AbstractArray, x::AbstractArray, p::Tuple)
    return (η_m .* x)
end


#  l2/3 excit, needs initial condition of itself
function fun_z_equ(y::AbstractArray, s::AbstractArray,  z_init::AbstractArray,  p::Tuple)
    return (λ .* relu.(y)) .+ imfilter(fun_F.(z_init, Γ), reflect(kern_H)) .+ (a_23_ex .* att) .- (ϕ .* imfilter(s, reflect(kern_T_p))) ./
    (1 .+ (λ .* relu.(y)) .+ imfilter(fun_F.(z_init, Γ), reflect(kern_H)) .+ (a_23_ex .* att) .+ imfilter(s, reflect(kern_T_p)))
    return
end


#  l2/3 excit - no feedback kernel
function fun_z_equ_noFb(y::AbstractArray, s::AbstractArray, p::Tuple)
    return (λ .* relu.(y)) .- (ϕ .* imfilter(s, reflect(kern_T_p))) ./
    (1 .+ (λ .* relu.(y)) .+ imfilter(s, reflect(kern_T_p)))
    return
end


#  l2/3 inhib, needs initial condition of itself
function fun_s_equ(z::AbstractArray, s::AbstractArray,  s_init::AbstractArray, p::Tuple)
    return ((imfilter(fun_F.(z, Γ), reflect(kern_H)) + a_23_in .* att ) ./ (1 .+ imfilter(s_init, reflect(kern_T_m))))           #????????? is T right?
end


#  l2/3 inhib - no feedback kernel
function fun_s_equ_noFb(z::AbstractArray, s::AbstractArray, p::Tuple)
    return (imfilter(fun_F.(z, Γ), reflect(kern_H)))
end


# V2 L6
function fun_xV2_equ(x_v2::AbstractArray, z::AbstractArray, z_v2::AbstractArray, p::Tuple)
    return fun_equ.((V_12_6 .* fun_F.(z, Γ) .+ (ϕ .* fun_F.(z_v2, Γ))))
end


#  V2 L4 excit
function fun_yV2_equ(z::AbstractArray, x::AbstractArray,  m::AbstractArray, p::Tuple)
    return (V_12_4 .* fun_F.(z,Γ) .+ (η_p .* x) - fun_f.(imfilter(m, reflect(kern_W_p)), μ, ν, n) ./
   (1 .+ V_12_4 .* fun_F.(z,Γ) .+ (η_p .* x) .+ fun_f.(imfilter(m, reflect(kern_W_p)), μ, ν, n)))
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
# # todo: lgn_A
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
# # todo: sTs
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
# # todo: sTs
# ds_v2 = δ_s(    -s_v2 +
#                 imfilter((fun_F(z_v2,Γ)),H_v2) -
#                 (s_v2 .* imfilter(s_v2, T_m)))  #?????


