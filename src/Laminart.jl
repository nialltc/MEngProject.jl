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

using NNlib, ImageFiltering, Images, OffsetArrays
# , MEngProject.LamKernels

export I_u, fun_v_C, fun_equ

function f!(du, u, p, t)
    @inbounds begin
        x = @view u[:, :, 1:p.K]
        y = @view u[:, :, p.K+1:2*p.K]
        m = @view u[:, :, 2*p.K+1:3*p.K]
        z = @view u[:, :, 3*p.K+1:4*p.K]
        s = @view u[:, :, 4*p.K+1:5*p.K]

#    C = @view uu[:, :, 5*p.K+1:6*p.K]
#    H_z = @view uu[:, :, 6*p.K+1:7*p.K]

        v_p = @view u[:, :, 5*p.K+1]
        v_m = @view u[:, :, 5*p.K+2]
#    x_lgn = @view uu[:, :, 7*p.K+3]

        dx = @view du[:, :, 1:p.K]
        dy = @view du[:, :, p.K+1:2*p.K]
        dm = @view du[:, :, 2*p.K+1:3*p.K]
        dz = @view du[:, :, 3*p.K+1:4*p.K]
        ds = @view du[:, :, 4*p.K+1:5*p.K]

        dv_p = @view du[:, :, 5*p.K+1]
        dv_m = @view du[:, :, 5*p.K+2]

        x_lgn = fun_x_lgn(x, p)
        C = fun_v_C(v_p, v_m, p)
        H_z = fun_H_z(z, p)

        temp_dv_p = fun_dv(v_p, p.r, x_lgn, p)
        temp_dv_m = fun_dv(v_m, .-p.r, x_lgn, p)
        temp_dx = fun_dx_V1(x, C, z, p.x_V2, p)
        temp_dy = fun_dy(y, C, x, m, p)
        temp_dm = fun_dm(m, x, p)
        temp_dz = fun_dz(z, y, H_z, s, p)
        temp_ds = fun_ds(s, z, H_z, p)

        @. dv_p = temp_dv_p
        @. dv_m = temp_dv_m
        @. dx = temp_dx
        @. dy = temp_dy
        @. dm = temp_dm
        @. dz = temp_dz
        @. ds = temp_ds
   end
   nothing
end


function kernels(img::AbstractArray, p::NamedTuple)
    C_A_temp = reshape(Array{eltype(img)}(undef, p.C_AB_l, p.C_AB_l * p.K), p.C_AB_l, p.C_AB_l, p.K)
    C_B_temp = copy(C_A_temp)
    H_temp = reshape(Array{eltype(img)}(undef, p.H_l, p.H_l * p.K), p.H_l, p.H_l, p.K)
    T_temp = reshape(Array{eltype(img)}(undef, 1, 1 * p.K), 1, 1, p.K)     #ijk,  1x1xk,   ijk
    W_temp = reshape(Array{eltype(img)}(undef, 19, 19 * p.K * p.K), 19, 19, p.K, p.K)     #ijk,  1x1xk,   ijk
    for k ∈ 1:p.K
        θ = π*(k-1)/p.K
        C_A_temp[:,:,k] = reflect(centered(LamKernels.kern_A(p.σ_2, θ)))           #ij ijk ijk
        C_B_temp[:,:,k] = reflect(centered(LamKernels.kern_B(p.σ_2, θ)))               #ij ijk ijk
        H_temp[:,:,k] = reflect(p.H_fact .* LamKernels.gaussian_rot(p.H_σ_x, p.H_σ_y, θ, p.H_l))  #ijk, ij for each k; ijk
        T_temp[:,:,k] = reshape([p.T_fact[k]],1,1)
#todo: generalise T and W for higher K
#         T_temp[:,:,k] = KernelFactors.gaussian(p.T_σ, p.K)
#         for l ∈ 1:p.K
#             W_temp[:,:,l,k] =
#         end
    end
    W_temp[:,:,1,1] = reflect(5 .* LamKernels.gaussian_rot(3,0.8,0,19) .+ LamKernels.gaussian_rot(0.4,1,0,19))
    W_temp[:,:,2,2] = reflect(5 .* LamKernels.gaussian_rot(3,0.8,0,19) .+ LamKernels.gaussian_rot(0.4,1,π/2,19))
    W_temp[:,:,1,2] = reflect(relu.(0.2 .- LamKernels.gaussian_rot(2,0.6,0,19) .- LamKernels.gaussian_rot(0.3,1.2,0,19)))
    W_temp[:,:,2,1] = reflect(relu.(0.2 .- LamKernels.gaussian_rot(2,0.6,0,19) .- LamKernels.gaussian_rot(0.3,1.2,π/2,19)))

# todo fix W kernel
#  W_temp[:,:,1,1] = reflect(LamKernels.gaussian_rot(3,0.8,0,19))
#     W_temp[:,:,2,2] = reflect(LamKernels.gaussian_rot(3,0.8,0,19))
#     W_temp[:,:,1,2] = reflect(LamKernels.gaussian_rot(3,0.8,0,19))
#     W_temp[:,:,2,1] = reflect(LamKernels.gaussian_rot(3,0.8,0,19))

# todo: fix range of W H
#     W_range = -(p.W_size-1)/2:(p.W_size-1)/2
#     H_range = -(p.H_size-1)/2:(p.H_size-1)/2
    W_range = -9:9
    H_range = -9:9

    temp_out = (
    k_gauss_1 = reflect(Kernel.gaussian(p.σ_1)),
    k_gauss_2 = reflect(Kernel.gaussian(p.σ_2)),
    k_C_A = C_A_temp,
    k_C_B = C_B_temp,
    k_W_p = OffsetArray(W_temp, W_range, W_range, 1:p.K, 1:p.K),
    k_W_m = OffsetArray(W_temp, W_range, W_range, 1:p.K, 1:p.K),
    k_H = OffsetArray(H_temp, H_range, H_range, 1:p.K),
    k_T_p = (T_temp),
    k_T_m = (p.T_p_m .* T_temp),
    k_T_p_v2 = (p.T_v2_fact .* T_temp),
    k_T_m_v2 = (p.T_v2_fact .* p.T_p_m .* T_temp),
    dim_i = size(img)[1],
    dim_j = size(img)[2],
    x_V2 = reshape(zeros(typeof(img[1,1]), size(img)[1], size(img)[2] * p.K), size(img)[1], size(img)[2], p.K))
    return merge(p, temp_out)
end


function varables(I::AbstractArray, p::NamedTuple)
   v_p = zeros(typeof(I[1,1]), size(I)[1], size(I)[2])
   v_m = copy(v_p)
   x_lgn = copy(v_p)
   x = reshape(zeros(typeof(I[1,1]), size(I)[1], size(I)[2] * p.K), size(I)[1], size(I)[2], p.K)
   y = copy(x)
   m = copy(x)
   z = copy(x)
   s = copy(x)
   C = copy(x)
   H_z = copy(x)
   x_V2 = copy(x)
   y_V2 = copy(x)
   m_V2 = copy(x)
   z_V2 = copy(x)
   s_V2 = copy(x)
   H_z_V2 = copy(x)
   return [v_p, v_m, x_lgn, x, y, m, z, s, C, H_z, x_V2, y_V2, m_V2, z_V2, s_V2, H_z_V2]
end


function variables_sep(I::AbstractArray, p::NamedTuple)
   v_p = zeros(typeof(I[1,1]), size(I)[1], size(I)[2])
   v_m = copy(v_p)
   x_lgn = copy(v_p)
   x = reshape(zeros(typeof(I[1,1]), size(I)[1], size(I)[2] * p.K), size(I)[1], size(I)[2], p.K)
   y = copy(x)
   m = copy(x)
   z = copy(x)
   s = copy(x)
   C = copy(x)
   H_z = copy(x)
   x_V2 = copy(x)
   y_V2 = copy(x)
   m_V2 = copy(x)
   z_V2 = copy(x)
   s_V2 = copy(x)
   H_z_V2 = copy(x)
   return v_p, v_m, x_lgn, x, y, m, z, s, C, H_z, x_V2, y_V2, m_V2, z_V2, s_V2, H_z_V2
end

function variables_dict(I::AbstractArray, p::NamedTuple)
   v_p = zeros(typeof(I[1,1]), size(I)[1], size(I)[2])
   v_m = copy(v_p)
   x_lgn = copy(v_p)
   x = reshape(zeros(typeof(I[1,1]), size(I)[1], size(I)[2] * p.K), size(I)[1], size(I)[2], p.K)
   y = copy(x)
   m = copy(x)
   z = copy(x)
   s = copy(x)
   C = copy(x)
   H_z = copy(x)
   x_V2 = copy(x)
   y_V2 = copy(x)
   m_V2 = copy(x)
   z_V2 = copy(x)
   s_V2 = copy(x)
   H_z_V2 = copy(x)
   return v_p, v_m, x_lgn, x, y, m, z, s, C, H_z, x_V2, y_V2, m_V2, z_V2, s_V2, H_z_V2
end

# function varables(I::AbstractArray, p::NamedTuple)
#    ij = zeros(typeof(I[1,1]), size(I)[1], size(I)[2])
#    ijk = reshape(zeros(typeof(I[1,1]), size(I)[1], size(I)[2] * p.K), size(I)[1], size(I)[2], p.K)
#    temp_out = [v_p = copy(ij),
#    v_m = copy(ij),
#    x_lgn = copy(ij),
#    x = copy(ijk),
#    y = copy(ijk),
#    m = copy(ijk),
#    z = copy(ijk),
#    s = copy(ijk),
#    C = copy(ijk),
#    H_z = copy(ijk),
#    x_V2 = copy(ijk),
#    y_V2 = copy(ijk),
#    m_V2 = copy(ijk),
#    z_V2 = copy(ijk),
#    s_V2 = copy(ijk),
#    H_z_V2 = copy(ijk)]
#    return temp_out
# end

function add_I_u_p(I::AbstractArray, p::NamedTuple)
    temp_out = (I = I, r = I_u(I, p))
    return merge(p, temp_out)
end


# retina

function I_u(I::AbstractArray, p::NamedTuple)
    return I - imfilter(I, p.k_gauss_1, p.filling)
end



# lgn feedback

# todo: test
# todo: should lgn_a be normalized, ie divide by k??

function fun_x_lgn(x::AbstractArray, p::NamedTuple)
   # todo: change to abstract array? or is eltype doing that??
#     x_lgn = Array{eltype(x)}(0, p.dim_i, p.dim_j)
    x_lgn = zeros(p.dim_i, p.dim_j)
#     todo: change to map function?
    for k in 1:p.K
        x_lgn += x[:,:,k]
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



function fun_F(value::Real, p::NamedTuple)
    max.(value - p.Γ, 0)
end


# williomson uses differnt F, relu with threshold
function fun_F_willimson(value::Real, p::NamedTuple)
    value < p.Γ ? zero(value) : value
end




# todo: check
function fun_f(x::AbstractArray, p::NamedTuple)
    return (p.μ .* x .^p.n) ./ (p.ν^p.n .+ x.^p.n)
end


function func_filter_W(img::AbstractArray, W::AbstractArray, p::NamedTuple)
#     out = reshape(fill(0/img[1,1,1], size(img)[1], size(img)[2] * p.K * p.K), size(img)[1], size(img)[2], p.K, p.K)
#     out = reshape(Array{eltype(img)}(undef, size(img)[1], size(img)[2] * p.K), size(img)[1], size(img)[2], p.K)
    out = copy(img)
    for k ∈ 1:p.K
#     todo fix W
        out[:,:,k] = imfilter(img[:,:,k], (centered(W[:,:,k,k]),), p.filling)
#         out[:,:,k] = img[:,:,k]
#         out[:,:,k] = conv(img[:,:,k], W[:,:,k,k])
        for l ∈ 1:p.K
            if l ≠ k
#                 out[:,:,k] += img[:,:,l]
                out[:,:,k] += imfilter(img[:,:,l], (centered(W[:,:,k,l]),), p.filling)
#                 out[:,:,k] .+= conv(img[:,:,l], W[:,:,k,l])
            end
        end
    end
    return out
end


# LGN
function fun_dv(v::AbstractArray, u::AbstractArray, x_lgn::AbstractArray, p::NamedTuple)
#     x_lgn = fun_x_lgn(x)
    return p.δ_v .* ( .- v .+
            ((1 .- v) .* max.(u,0) .* (1 .+ p.C_1 .* x_lgn)) .-
            ((1 .+ v) .* p.C_2 .* imfilter(x_lgn, (centered(p.k_gauss_1),), p.filling)))
#             ((1 .+ v) .* p.C_2 .* conv(x_lgn, p.k_gauss_1)))
end


# lgn to l6 and l4
function fun_v_C(v_p::AbstractArray, v_m::AbstractArray, p::NamedTuple)
    # isodd(l) || throw(ArgumentError("length must be odd"))

    V = exp(-1/8) .* (imfilter((max.(v_p,0)-max.(v_m,0)), (centered(p.k_gauss_2),), p.filling))
#     V = exp(-1/8) .* (conv((max.(v_p,0) .- max.(v_m,0)), p.k_gauss_2))

# todo: change to abstract array? or is eltype doing that??
    A = reshape(Array{eltype(V)}(undef, size(V)[1], size(V)[2]*p.K),size(V)[1],size(V)[2],p.K)
    B = copy(A)

    for k in 1:p.K
#         θ = π*(k-1)/p.K
        A[:,:,k] = imfilter(V, (centered(p.k_C_A[:,:,k]),), p.filling)
        B[:,:,k] = abs.(imfilter(V, (centered(p.k_C_B[:,:,k]),), p.filling))
#         A[:,:,k] = conv(V, p.k_C_A[:,:,k])
#         B[:,:,k] = abs.(conv(V, p.k_C_B[:,:,k]))
    end

    return p.γ .* (max.(A .- B, 0) .+ max.(.- A .- B, 0))
end



# L6
function fun_dx_V1(x::AbstractArray, C::AbstractArray, z::AbstractArray, x_v2::AbstractArray, p::NamedTuple)
    return p.δ_c .* (-x .+
            ((1 .- x) .*
                ((p.α*C) .+ (p.ϕ .* max.(z .- p.Γ,0)) .+ (p.V_21 .* x_v2) .+ p.att)))
end



#     L4 excit
function fun_dy(y::AbstractArray, C::AbstractArray, x::AbstractArray, m::AbstractArray, p::NamedTuple)
    return  p.δ_c .* (   .-y .+
            ((1 .- y) .* (C .+ (p.η_p .* x))) .-
            ((1 .+ y) .* fun_f(m .* func_filter_W(m, p.k_W_p, p), p)))
end

# function fun_dy(y::AbstractArray, C::AbstractArray, x::AbstractArray, m::AbstractArray, p::NamedTuple)
#     return  p.δ_c .* (   .-y .+
#             ((1 .- y) .* (C .+ (p.η_p .* x))) .-
#             ((1 .+ y) .* fun_f(m , p)))
# end


#     l4 inhib
function fun_dm(m::AbstractArray, x::AbstractArray, p::NamedTuple)
    return p.δ_m .* (  -m .+
                     (p.η_m .* x) .-
                     (m .* fun_f(func_filter_W(m, p.k_W_m, p), p)))
end

# function fun_dm(m::AbstractArray, x::AbstractArray, p::NamedTuple)
#     return p.δ_m .* (  -m .+
#                      (p.η_m .* x) .-
#                      (m .* fun_f(m, p)))
# end


#     L2/3 excit
function fun_dz(z::AbstractArray, y::AbstractArray,  H_z::AbstractArray, s::AbstractArray, p::NamedTuple)
    return p.δ_z .*   (-z .+
                    ((1 .- z) .*
#                         ((p.λ .* max.(y,0)) .+ H_z )) .-
                        ((p.λ .* max.(y,0)) .+ H_z .+ (p.a_23_ex .* p.att))) .-
                    ((z .+ p.ψ) .* (imfilter(s, (centered(p.k_T_p),), p.filling))))
#                     ((z .+ p.ψ) .* (conv(s, p.k_T_p))))
end



#     L2/3 inhib
function fun_ds(s::AbstractArray, z::AbstractArray, H_z::AbstractArray, p::NamedTuple)
    return p.δ_s .*   ( -s .+
                    H_z .+ (p.a_23_in .* p.att) .-
                    (s .* imfilter(s, (centered(p.k_T_m),), p.filling)))  #?????
#                     (s .* conv(s, p.k_T_m)))  #?????
end

function fun_H_z(z::AbstractArray, p::NamedTuple)
    H_z_out = copy(z)
    for k ∈ 1:p.K
#     fix H_z temp!!
#         H_z_out[:,:,k] = imfilter((max.(z[:,:,k] .- p.Γ,0)), (centered(p.k_H[:,:,k]),), p.filling)
        H_z_out[:,:,k] = max.(z[:,:,k] .- p.Γ,0)
#         H_z_out[:,:,k] = conv((max.(z[:,:,k] .- p.Γ,0)), p.k_H[:,:,k])
        end
        return H_z_out
end



#     V2 L6
function fun_dx_v2(x_v2::AbstractArray, z_v2::AbstractArray, z::AbstractArray, p::NamedTuple)
    return p.δ_c .*   (  -x_v2 .+
                    ((1 .- x_v2) .*
                        ((p.v_12_6 .* max.(z .- p.Γ, 0)) + (p.ϕ .* max.(z_v2 .- p.Γ, 0)) .+ p.att)))
end


# V2 L4 excit
function fun_dy_v2(y_v2::AbstractArray, z::AbstractArray, x_v2::AbstractArray, m_v2::AbstractArray, p::NamedTuple)
    return δ_c .*   (-y_v2 .+
                    ((1 .- y_v2) .* ((v_12_4 .* max.(z .- p.Γ, 0)) .+ (p.η_p .* x_v2))) .-
                    ((1 .+ y_v2) .* fun_f.(imfilter(m_v2, (centered(p.k_W_p),), p.filling)), p.μ, p.ν, p.n))
#                     ((1 .+ y_v2) .* fun_f.(conv(m_v2, p.k_W_p)), p.μ, p.ν, p.n))
end



# for equilabrium
fun_equ(x) = x/(1+x)


# lgn
function fun_v_equ(u::AbstractArray, x::AbstractArray, p::NamedTuple)
    return fun_equ.(relu.(u) .* (p.lgn_equ_u .+ (p.lgn_equ_A .* fun_A.(x))) .- (p.lgn_equ_B .* fun_B.(x)))
end


# lgn, no L6 feedback, light
function fun_v_equ_noFb(u::AbstractArray, x::AbstractArray, p::NamedTuple)
    return fun_equ.(relu.(u) .* (p.lgn_para_u))
end


# l6
function fun_x_equ(C::AbstractArray, z::AbstractArray,  p::NamedTuple)
    return fun_equ.((p.α .* C) .+ (p.ϕ .* fun_F.(z,p.Γ)) + (p.V_21 .* x_v2) .+ p.att)
end


# l6, no V2 feedback, light
function fun_x_equ_noV2(C::AbstractArray, z::AbstractArray, p::NamedTuple)
    return fun_equ.((p.α .* C) .+ (p.ϕ .* fun_F.(z,p.Γ)))
end


# l4 excit
function fun_y_equ(C::AbstractArray, x::AbstractArray,  m::AbstractArray, p::NamedTuple)
    return fun_equ.(C .+ (η_p .* x - fun_f.(imfilter(m, p.k_W_p), p.μ, p.ν, p.n)))
end

# l4 inhib, needs initial condition of itself
function fun_m_equ(C::AbstractArray, x::AbstractArray, m_init::AbstractArray, p::NamedTuple)
    return (p.η_m .* x ./ (1 .+  fun_f.(imfilter(m_init, p.k_W_m), p.μ, p.ν, p.n)))
end


# l4 inhib - no feedback kernel
function fun_m_equ_noFb(C::AbstractArray, x::AbstractArray, p::NamedTuple)
    return (p.η_m .* x)
end


#  l2/3 excit, needs initial condition of itself
function fun_z_equ(y::AbstractArray, s::AbstractArray,  z_init::AbstractArray,  p::NamedTuple)
    return (p.λ .* relu.(y)) .+ imfilter(fun_F.(z_init, p.Γ), k_H) .+ (p.a_23_ex .* p.att) .- (p.ϕ .* imfilter(s, k_T_p)) ./
    (1 .+ (p.λ .* relu.(y)) .+ imfilter(fun_F.(z_init, p.Γ), k_H) .+ (p.a_23_ex .* p.att) .+ imfilter(s, k_T_p))
    return
end


#  l2/3 excit - no feedback kernel
function fun_z_equ_noFb(y::AbstractArray, s::AbstractArray, p::NamedTuple)
    return (p.λ .* relu.(y)) .- (p.ϕ .* imfilter(s, k_T_p)) ./
    (1 .+ (p.λ .* relu.(y)) .+ imfilter(s, k_T_p))
    return
end


#  l2/3 inhib, needs initial condition of itself
function fun_s_equ(z::AbstractArray, s::AbstractArray,  s_init::AbstractArray, p::NamedTuple)
    return ((imfilter(fun_F.(z, p.Γ), k_H) + p.a_23_in .* p.att ) ./ (1 .+ imfilter(s_init, k_T_m)))           #????????? is T right?
end


#  l2/3 inhib - no feedback kernel
function fun_s_equ_noFb(z::AbstractArray, s::AbstractArray, p::NamedTuple)
    return (imfilter(fun_F.(z, p.Γ), k_H))
end


# V2 L6
function fun_xV2_equ(x_v2::AbstractArray, z::AbstractArray, z_v2::AbstractArray, p::NamedTuple)
    return fun_equ.((p.V_12_6 .* fun_F.(z, p.Γ) .+ (p.ϕ .* fun_F.(z_v2, p.Γ))))
end


#  V2 L4 excit
function fun_yV2_equ(z::AbstractArray, x::AbstractArray,  m::AbstractArray, p::NamedTuple)
    return (p.V_12_4 .* fun_F.(z,p.Γ) .+ (p.η_p .* x) - fun_f.(imfilter(m, k_W_p), p.μ, p.ν, p.n) ./
   (1 .+ p.V_12_4 .* fun_F.(z,p.Γ) .+ (p.η_p .* x) .+ fun_f.(imfilter(m, k_W_p), p.μ, p.ν, p.n)))
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
# lgn_B = C_2 * imfilter(x, Kernel.gaussian(σ_1), p.filling)

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


