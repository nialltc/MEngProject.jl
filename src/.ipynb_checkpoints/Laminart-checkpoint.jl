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

struct MyFunction{T} <: Function
  x_lgn::T
  C::T
  H_z::T
end

function (ff::MyFunction)(du, u, p, t)
# function f!(du, u, p, t)
    @inbounds begin
        x = @view u[:, :, 1:p.K]
        y = @view u[:, :, p.K+1:2*p.K]
        m = @view u[:, :, 2*p.K+1:3*p.K]
        z = @view u[:, :, 3*p.K+1:4*p.K]
        s = @view u[:, :, 4*p.K+1:5*p.K]

        #    C = @view u[:, :, 5*p.K+1:6*p.K]
        #    H_z = @view u[:, :, 6*p.K+1:7*p.K]

        v_p = @view u[:, :, 5*p.K+1]
        v_m = @view u[:, :, 5*p.K+2]
        #    x_lgn = @view u[:, :, 7*p.K+3]

        dx = @view du[:, :, 1:p.K]
        dy = @view du[:, :, p.K+1:2*p.K]
        dm = @view du[:, :, 2*p.K+1:3*p.K]
        dz = @view du[:, :, 3*p.K+1:4*p.K]
        ds = @view du[:, :, 4*p.K+1:5*p.K]

        dv_p = @view du[:, :, 5*p.K+1]
        dv_m = @view du[:, :, 5*p.K+2]

        x_lgn = @view ff.x_lgn[:,:,1]
        #         x_lgn = similar(v_p)
        #         C = similar(x)
        #         H_z = similar(x)
        # x_lgn = Array{eltype(u)}(undef, p.dim_i, p.dim_j)
        #         C = reshape(Array{eltype(u)}(undef, p.dim_i, p.dim_j*p.K),p.dim_i,p.dim_j, p.K)
        #         C = reshape(zeros(p.dim_i, p.dim_j*p.K),p.dim_i,p.dim_j, p.K)
        # C = copy(u[:, :, 1:p.K])
        # H_z = copy(u[:, :, 1:p.K])

        fun_x_lgn!(x_lgn, x, p)
        fun_v_C!(ff.C, v_p, v_m, p)
        fun_H_z!(ff.H_z, z, p)

        fun_dv!(dv_p, v_p, p.r, x_lgn, p)
        fun_dv!(dv_m, v_m, .-p.r, x_lgn, p)
        fun_dx_v1!(dx, x, ff.C, z, p.x_V2, p)
        fun_dy!(dy, y, ff.C, x, m, p)
        fun_dm!(dm, m, x, p)
        fun_dz!(dz, z, y, ff.H_z, s, p)
        fun_ds!(ds, s, ff.H_z, p)

    end
    nothing
end


function kernels(img::AbstractArray, p::NamedTuple)
    C_A_temp = reshape(
        Array{eltype(img)}(undef, p.C_AB_l, p.C_AB_l * p.K),
        p.C_AB_l,
        p.C_AB_l,
        p.K,
    )
    C_B_temp = copy(C_A_temp)
    H_temp = reshape(
        Array{eltype(img)}(undef, p.H_l, p.H_l * p.K),
        p.H_l,
        p.H_l,
        p.K,
    )
    T_temp = reshape(Array{eltype(img)}(undef, 1, 1 * p.K), 1, 1, p.K)     #ijk,  1x1xk,   ijk
    W_temp =
        reshape(Array{eltype(img)}(undef, p.W_l, p.W_l * p.K * p.K), p.W_l, p.W_l, p.K, p.K)     #ijk,  1x1xk,   ijk
    for k ∈ 1:p.K
        θ = π * (k - 1) / p.K
        C_A_temp[:, :, k] = reflect(centered(LamKernels.kern_d(p.σ_2, θ)))           #ij ijk ijk
        C_B_temp[:, :, k] = reflect(centered(LamKernels.kern_b(p.σ_2, θ)))               #ij ijk ijk
        H_temp[:, :, k] = reflect(
            p.H_fact .* LamKernels.gaussian_rot(p.H_σ_x, p.H_σ_y, θ, p.H_l),
        )  #ijk, ij for each k; ijk
        T_temp[:, :, k] = reshape([p.T_fact[k]], 1, 1)
        #todo: generalise T and W for higher K
        #         T_temp[:,:,k] = KernelFactors.gaussian(p.T_σ, p.K)
        #         for l ∈ 1:p.K
        #             W_temp[:,:,l,k] =
        #         end
    end
    W_temp[:, :, 1, 1] = reflect(
        5 .* LamKernels.gaussian_rot(3, 0.8, 0, p.W_l) .+
        LamKernels.gaussian_rot(0.4, 1, 0, p.W_l),
    )
    W_temp[:, :, 2, 2] = reflect(
        5 .* LamKernels.gaussian_rot(3, 0.8, π / 2, p.W_l) .+
        LamKernels.gaussian_rot(0.4, 1, π / 2, p.W_l),
    )
    W_temp[:, :, 1, 2] = reflect(relu.(
        0.2 .- LamKernels.gaussian_rot(2, 0.6, 0, p.W_l) .-
        LamKernels.gaussian_rot(0.3, 1.2, 0, p.W_l),
    ))
    W_temp[:, :, 2, 1] = reflect(relu.(
        0.2 .- LamKernels.gaussian_rot(2, 0.6, π / 2, p.W_l) .-
        LamKernels.gaussian_rot(0.3, 1.2, π / 2, p.W_l),
    ))

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
        k_W_p = W_temp,
        k_W_m = W_temp,
        # k_W_m = OffsetArray(W_temp, W_range, W_range, 1:p.K, 1:p.K),
        k_H = H_temp,
        # k_H = OffsetArray(H_temp, H_range, H_range, 1:p.K),
        k_T_p = T_temp,
        k_T_m = (p.T_p_m .* T_temp),
        k_T_p_v2 = (p.T_v2_fact .* T_temp),
        k_T_m_v2 = (p.T_v2_fact .* p.T_p_m .* T_temp),
        dim_i = size(img)[1],
        dim_j = size(img)[2],
        x_V2 = reshape(
            zeros(typeof(img[1, 1]), size(img)[1], size(img)[2] * p.K),
            size(img)[1],
            size(img)[2],
            p.K,
        ),
    )
    return merge(p, temp_out)
end


# function varables(I::AbstractArray, p::NamedTuple)
#    v_p = zeros(typeof(I[1,1]), size(I)[1], size(I)[2])
#    v_m = copy(v_p)
#    x_lgn = copy(v_p)
#    x = reshape(zeros(typeof(I[1,1]), size(I)[1], size(I)[2] * p.K), size(I)[1], size(I)[2], p.K)
#    y = copy(x)
#    m = copy(x)
#    z = copy(x)
#    s = copy(x)
#    C = copy(x)
#    H_z = copy(x)
#    x_V2 = copy(x)
#    y_V2 = copy(x)
#    m_V2 = copy(x)
#    z_V2 = copy(x)
#    s_V2 = copy(x)
#    H_z_V2 = copy(x)
#    return [v_p, v_m, x_lgn, x, y, m, z, s, C, H_z, x_V2, y_V2, m_V2, z_V2, s_V2, H_z_V2]
# end
#
#
# function variables_sep(I::AbstractArray, p::NamedTuple)
#    v_p = zeros(typeof(I[1,1]), size(I)[1], size(I)[2])
#    v_m = copy(v_p)
#    x_lgn = copy(v_p)
#    x = reshape(zeros(typeof(I[1,1]), size(I)[1], size(I)[2] * p.K), size(I)[1], size(I)[2], p.K)
#    y = copy(x)
#    m = copy(x)
#    z = copy(x)
#    s = copy(x)
#    C = copy(x)
#    H_z = copy(x)
#    x_V2 = copy(x)
#    y_V2 = copy(x)
#    m_V2 = copy(x)
#    z_V2 = copy(x)
#    s_V2 = copy(x)
#    H_z_V2 = copy(x)
#    return v_p, v_m, x_lgn, x, y, m, z, s, C, H_z, x_V2, y_V2, m_V2, z_V2, s_V2, H_z_V2
# end
#
# function variables_dict(I::AbstractArray, p::NamedTuple)
#    v_p = zeros(typeof(I[1,1]), size(I)[1], size(I)[2])
#    v_m = copy(v_p)
#    x_lgn = copy(v_p)
#    x = reshape(zeros(typeof(I[1,1]), size(I)[1], size(I)[2] * p.K), size(I)[1], size(I)[2], p.K)
#    y = copy(x)
#    m = copy(x)
#    z = copy(x)
#    s = copy(x)
#    C = copy(x)
#    H_z = copy(x)
#    x_V2 = copy(x)
#    y_V2 = copy(x)
#    m_V2 = copy(x)
#    z_V2 = copy(x)
#    s_V2 = copy(x)
#    H_z_V2 = copy(x)
#    return v_p, v_m, x_lgn, x, y, m, z, s, C, H_z, x_V2, y_V2, m_V2, z_V2, s_V2, H_z_V2
# end

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


function fun_x_lgn!(x_lgn::AbstractArray, x::AbstractArray, p::NamedTuple)
    @. x_lgn = 0.0
    # @inbounds begin
        for k ∈ 1:p.K
            @. x_lgn += @view x[:, :, k]
        # end
    end
    return nothing
end


# function fun_F(value::Real, p::NamedTuple)
#     max.(value - p.Γ, 0)
# end
#
#
# # williomson uses differnt F, relu with threshold
# function fun_F_willimson(value::Real, p::NamedTuple)
#     value < p.Γ ? zero(value) : value
# end




function fun_f!(f_out::AbstractArray, x::AbstractArray, p::NamedTuple)
    @. f_out = (p.μ * x^p.n) / (p.ν^p.n + x^p.n)
    return nothing
end



function func_filter_W!(
    W_out::AbstractArray,
    img::AbstractArray,
    W::AbstractArray,
    p::NamedTuple,
)
    temp_k = similar(W_out[:, :, 1])
    # @inbounds begin
        for k ∈ 1:p.K
            #     todo fix W
            img_k = @view img[:, :, k]
            out_k = @view W_out[:, :, k]
            imfilter!(out_k, img_k, centered(W[:, :, k, k]), p.filling)
            for l ∈ 1:p.K
                if l ≠ k
                    img_l = @view img[:, :, l]
                    imfilter!(
                        temp_k,
                        img_l,
                        centered(W[:, :, k, l]),
                        p.filling,
                    )
                    @. out_k += temp_k
                end
            end
        end
    # end
    return nothing
end



# LGN
function fun_dv!(
    dv::AbstractArray,
    v::AbstractArray,
    u::AbstractArray,
    x_lgn::AbstractArray,
    p::NamedTuple,
)
    imfilter!(dv, x_lgn, centered(p.k_gauss_1), p.filling)
    @. dv =
        p.δ_v * (
            -v + ((1f0 - v) * max(u, 0f0) * (1f0 + p.C_1 * x_lgn)) -
            ((1f0 + v) * p.C_2 * dv)
        )
    return nothing
end


# lgn to l6 and l4
function fun_v_C!(
    v_C::AbstractArray,
    v_p::AbstractArray,
    v_m::AbstractArray,
    p::NamedTuple,
)
    V = similar(v_p)
    temp = similar(v_p)

    @. temp = exp(-1.0f0 / 8.0f0) * (max(v_p, 0) - max(v_m, 0))
    imfilter!(V, temp, centered(p.k_gauss_2), p.filling)

    A = similar(v_C)
    #     allocate B to v_C
    # @inbounds begin
        for k ∈ 1:p.K
            a = @view A[:, :, k]
            b = @view v_C[:, :, k]
            imfilter!(a, V, centered(p.k_C_A[:, :, k]), p.filling)
            imfilter!(b, V, centered(p.k_C_B[:, :, k]), p.filling)
        end
    # end
    @. v_C = p.γ * (max(A - abs(v_C), 0) + max(-A - abs(v_C), 0))
    return nothing
end



# L6
function fun_dx_v1!(
    dx::AbstractArray,
    x::AbstractArray,
    C::AbstractArray,
    z::AbstractArray,
    x_v2::AbstractArray,
    p::NamedTuple,
)
    @. dx =
        p.δ_c * (
            -x + (
                (1.0f0 - x) * (
                    (p.α * C) + (p.ϕ * max(z - p.Γ, 0)) .+ (p.v_21 * x_v2) +
                    p.att
                )
            )
        )
    return nothing
end



#     L4 excit
function fun_dy!(
    dy::AbstractArray,
    y::AbstractArray,
    C::AbstractArray,
    x::AbstractArray,
    m::AbstractArray,
    p::NamedTuple,
)
    func_filter_W!(dy, m, p.k_W_p, p)
    @. dy = m * dy
    fun_f!(dy, dy, p)
    @. dy = p.δ_c * (-y + ((1 - y) * (C + (p.η_p * x))) - ((1 + y) * dy))
    return nothing
end


#     l4 inhib
function fun_dm!(
    dm::AbstractArray,
    m::AbstractArray,
    x::AbstractArray,
    p::NamedTuple,
)
    func_filter_W!(dm, m, p.k_W_m, p)
    fun_f!(dm, dm, p)
    @. dm = p.δ_m * (-m + (p.η_m * x) - (m * dm))
    return nothing
end



#     L2/3 excit
function fun_dz!(
    dz::AbstractArray,
    z::AbstractArray,
    y::AbstractArray,
    H_z::AbstractArray,
    s::AbstractArray,
    p::NamedTuple,
)
    imfilter!(dz, s, centered(p.k_T_p), p.filling)
    @. dz =
        p.δ_z * (
            -z + ((1 - z) * ((p.λ * max(y, 0)) + H_z + (p.a_23_ex * p.att))) -
            ((z + p.ψ) * dz)
        )
    return nothing
end

#     L2/3 inhib
function fun_ds!(
    ds::AbstractArray,
    s::AbstractArray,
    H_z::AbstractArray,
    p::NamedTuple,
)
    imfilter!(ds, s, centered(p.k_T_m), p.filling)
    @. ds = p.δ_s * (-s + H_z + (p.a_23_in * p.att) - (s * ds))
    return nothing
end


function fun_H_z!(H_z::AbstractArray, z::AbstractArray, p::NamedTuple)
    temp = similar(z)
    @. temp = max(z - p.Γ, 0)
    # @inbounds begin
        for k ∈ 1:p.K
            H_z_k = @view H_z[:, :, k]
            temp_k = @view temp[:, :, k]
            imfilter!(H_z_k, temp_k, centered(p.k_H[:, :, k]), p.filling)
        end
    # end
    return nothing
end



#     V2 L6
function fun_dx_v2!(
    dx::AbstractArray,
    x_v2::AbstractArray,
    z_v2::AbstractArray,
    z::AbstractArray,
    p::NamedTuple,
)
    @. dx =
        p.δ_c * (
            -x_v2 + (
                (1 - x_v2) * (
                    (p.v12_6 * max(z - p.Γ, 0)) +
                    (p.ϕ * max(z_v2 - p.Γ, 0)) +
                    p.att
                )
            )
        )
    return nothing
end


# V2 L4 excit
function fun_dy_v2!(
    dy::AbstractArray,
    y_v2::AbstractArray,
    z::AbstractArray,
    x_v2::AbstractArray,
    m_v2::AbstractArray,
    p::NamedTuple,
)
    func_filter_W!(dy, m_v2, p.k_W_p, p)
    fun_f!(dy, dy, p)
    @. dy =
        p.δ_c * (
            -y_v2 +
            ((1.0f0 - y_v2) * ((p.v12_4 * max(z - p.Γ, 0)) + (p.η_p * x_v2))) -
            ((1.0f0 + y_v2) * dy)
        )
    return nothing
end



# # for equilabrium
# fun_equ(x) = x/(1+x)
#
#
# # lgn
# function fun_v_equ(u::AbstractArray, x::AbstractArray, p::NamedTuple)
#     return fun_equ.(relu.(u) .* (p.lgn_equ_u .+ (p.lgn_equ_A .* fun_A.(x))) .- (p.lgn_equ_B .* fun_B.(x)))
# end
#
#
# # lgn, no L6 feedback, light
# function fun_v_equ_noFb(u::AbstractArray, x::AbstractArray, p::NamedTuple)
#     return fun_equ.(relu.(u) .* (p.lgn_para_u))
# end
#
#
# # l6
# function fun_x_equ(C::AbstractArray, z::AbstractArray,  p::NamedTuple)
#     return fun_equ.((p.α .* C) .+ (p.ϕ .* fun_F.(z,p.Γ)) + (p.V_21 .* x_v2) .+ p.att)
# end
#
#
# # l6, no V2 feedback, light
# function fun_x_equ_noV2(C::AbstractArray, z::AbstractArray, p::NamedTuple)
#     return fun_equ.((p.α .* C) .+ (p.ϕ .* fun_F.(z,p.Γ)))
# end
#
#
# # l4 excit
# function fun_y_equ(C::AbstractArray, x::AbstractArray,  m::AbstractArray, p::NamedTuple)
#     return fun_equ.(C .+ (η_p .* x - fun_f.(imfilter(m, p.k_W_p), p.μ, p.ν, p.n)))
# end
#
# # l4 inhib, needs initial condition of itself
# function fun_m_equ(C::AbstractArray, x::AbstractArray, m_init::AbstractArray, p::NamedTuple)
#     return (p.η_m .* x ./ (1 .+  fun_f.(imfilter(m_init, p.k_W_m), p.μ, p.ν, p.n)))
# end
#
#
# # l4 inhib - no feedback kernel
# function fun_m_equ_noFb(C::AbstractArray, x::AbstractArray, p::NamedTuple)
#     return (p.η_m .* x)
# end
#
#
# #  l2/3 excit, needs initial condition of itself
# function fun_z_equ(y::AbstractArray, s::AbstractArray,  z_init::AbstractArray,  p::NamedTuple)
#     return (p.λ .* relu.(y)) .+ imfilter(fun_F.(z_init, p.Γ), k_H) .+ (p.a_23_ex .* p.att) .- (p.ϕ .* imfilter(s, k_T_p)) ./
#     (1 .+ (p.λ .* relu.(y)) .+ imfilter(fun_F.(z_init, p.Γ), k_H) .+ (p.a_23_ex .* p.att) .+ imfilter(s, k_T_p))
#     return
# end
#
#
# #  l2/3 excit - no feedback kernel
# function fun_z_equ_noFb(y::AbstractArray, s::AbstractArray, p::NamedTuple)
#     return (p.λ .* relu.(y)) .- (p.ϕ .* imfilter(s, k_T_p)) ./
#     (1 .+ (p.λ .* relu.(y)) .+ imfilter(s, k_T_p))
#     return
# end
#
#
# #  l2/3 inhib, needs initial condition of itself
# function fun_s_equ(z::AbstractArray, s::AbstractArray,  s_init::AbstractArray, p::NamedTuple)
#     return ((imfilter(fun_F.(z, p.Γ), k_H) + p.a_23_in .* p.att ) ./ (1 .+ imfilter(s_init, k_T_m)))           #????????? is T right?
# end
#
#
# #  l2/3 inhib - no feedback kernel
# function fun_s_equ_noFb(z::AbstractArray, s::AbstractArray, p::NamedTuple)
#     return (imfilter(fun_F.(z, p.Γ), k_H))
# end
#
#
# # V2 L6
# function fun_xV2_equ(x_v2::AbstractArray, z::AbstractArray, z_v2::AbstractArray, p::NamedTuple)
#     return fun_equ.((p.V_12_6 .* fun_F.(z, p.Γ) .+ (p.ϕ .* fun_F.(z_v2, p.Γ))))
# end
#
#
# #  V2 L4 excit
# function fun_yV2_equ(z::AbstractArray, x::AbstractArray,  m::AbstractArray, p::NamedTuple)
#     return (p.V_12_4 .* fun_F.(z,p.Γ) .+ (p.η_p .* x) - fun_f.(imfilter(m, k_W_p), p.μ, p.ν, p.n) ./
#    (1 .+ p.V_12_4 .* fun_F.(z,p.Γ) .+ (p.η_p .* x) .+ fun_f.(imfilter(m, k_W_p), p.μ, p.ν, p.n)))
# end

end
