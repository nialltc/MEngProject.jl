"""
# module LaminartEqImfilter

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-08-20

LAMINART equations from

R. D. S. Raizada and S. Grossberg, ‘Context-sensitive binding by the laminar circuits of V1 and V2: A unified model of perceptual grouping, attention, and orientation contrast’, Visual Cognition, vol. 8, no. 3–5, p. 431, Jun. 2001, doi: 10/d8phhc.

Deprecated version
"""

module LaminartEqImfilter_old

using NNlib, ImageFiltering, Images, OffsetArrays


# retina

function I_u!(r::AbstractArray, I::AbstractArray, p::NamedTuple)
    imfilter!(r, I, p.k_gauss_1, p.filling)
    @. r = I - r
    return nothing
end


# lgn feedback


function fun_x_lgn!(x_lgn::AbstractArray, x::AbstractArray, p::NamedTuple)
    @. x_lgn = 0.0f0
    # @inbounds begin
    for k ∈ 1:p.K
        @. x_lgn += @view x[:, :, k]
        # end
    end
    return nothing
end


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
                imfilter!(temp_k, img_l, centered(W[:, :, k, l]), p.filling)
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

    @. temp = exp(-1.0f0 / 8.0f0) * (max(v_p, 0f0) - max(v_m, 0f0))
    imfilter!(V, temp, centered(p.k_gauss_2), p.filling)

    A = similar(v_C)
    #     allocate B to v_C
    # @inbounds begin
    for k ∈ 1:p.K
        a = @view A[:, :, k]
        b = @view v_C[:, :, k]
        imfilter!(a, V, centered(p.k_C_d[:, :, k]), p.filling)
        imfilter!(b, V, centered(p.k_C_b[:, :, k]), p.filling)
    end
    # end
    @. v_C = p.γ * (max(A - abs(v_C), 0f0) + max(-A - abs(v_C), 0f0))
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
                    (p.α * C) + (p.ϕ * max(z - p.Γ, 0f0)) .+ (p.v_21 * x_v2) +
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
    @. dy = p.δ_c * (-y + ((1f0 - y) * (C + (p.η_p * x))) - ((1f0 + y) * dy))
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
            -z +
            ((1f0 - z) * ((p.λ * max(y, 0f0)) + H_z + (p.a_23_ex * p.att))) -
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
    @. temp = max(z - p.Γ, 0f0)
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
                (1f0 - x_v2) * (
                    (p.v12_6 * max(z - p.Γ, 0f0)) +
                    (p.ϕ * max(z_v2 - p.Γ, 0f0)) +
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
            -y_v2 + (
                (1.0f0 - y_v2) *
                ((p.v12_4 * max(z - p.Γ, 0f0)) + (p.η_p * x_v2))
            ) - ((1.0f0 + y_v2) * dy)
        )
    return nothing
end


# # # for equilabrium


# # l6 equilabrium
function fun_x_equ!(
    x::AbstractArray,
    C::AbstractArray,
    z::AbstractArray,
    x_v2::AbstractArray,
    x_temp::AbstractArray,
    p::NamedTuple,
)
    # 	@inbounds begin
    @. x_temp = (p.α * C) + (p.ϕ * max(z - p.Γ, 0f0)) + (p.v_21 * x_v2) + p.att
    @. x = x_temp / (1f0 + x_temp)
    # 	end
    return nothing
end


# # l4 excit equilabrium
function fun_y_equ!(
    y::AbstractArray,
    C::AbstractArray,
    x::AbstractArray,
    m::AbstractArray,
    dy_temp::AbstractArray,
    p::NamedTuple,
)
    # 	@inbounds begin
    imfilter!(dy_temp, m, p.k_W_p, p.filling)
    @. dy_temp = m * dy_temp
    fun_f!(dy_temp, p)
    @. y = (C + (p.η_p * x) - dy_temp) / (1f0 + C + (p.η_p * x) + dy_temp)
    # 	end
    return nothing
end


end
