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

module LaminartEqConv


using NNlib, ImageFiltering, Images, OffsetArrays, CUDA



function conv!(out::AbstractArray, img::AbstractArray, kern::AbstractArray, p::NamedTuple)
    @inbounds NNlib.conv!(out, img, kern, NNlib.DenseConvDims(img, kern, padding = ( size(kern)[1]>>1 , size(kern)[1]>>1 , size(kern)[2]>>1 , size(kern)[2]>>1 ), flipkernel = true))
    return nothing
end


function fun_f!(arr::AbstractArray, p::NamedTuple)
   @inbounds @. arr = (p.μ * arr^p.n) / (p.ν_pw_n + arr^p.n)
    return nothing
end


# retina

function I_u!(r::AbstractArray, I::AbstractArray, p::NamedTuple)
    conv!(r, I, p.k_gauss_1, p)
    @. r = I - r
end



# LGN
function fun_dv!(
    dv::AbstractArray,
    v::AbstractArray,
    u::AbstractArray,
    x_lgn::AbstractArray,
    dv_temp::AbstractArray,
    p::NamedTuple,
)
    @inbounds begin
    conv!(dv_temp, x_lgn, p.k_gauss_1, p)
    @. dv =
        p.δ_v * (
            -v + ((1f0 - v) * max(u, 0f0) * (1f0 + p.C_1 * x_lgn)) -
            ((1f0 + v) * p.C_2 * dv_temp)
        )
    end
    return nothing
end


function fun_x_lgn!(x_lgn::AbstractArray, x::AbstractArray, p::NamedTuple)
    @inbounds NNlib.conv!(x_lgn, x, p.k_x_lgn, NNlib.DenseConvDims(x, p.k_x_lgn, padding = 0, flipkernel = true))
    return nothing
end


function fun_v_C!(
    v_C::AbstractArray,
    v_p::AbstractArray,
    v_m::AbstractArray, 
    V_temp_1::AbstractArray,
    V_temp_2::AbstractArray,
    A_temp::AbstractArray,
    B_temp::AbstractArray,
    p::NamedTuple,
)

    @inbounds begin
        @. V_temp_2 = exp(-1.0f0 / 8.0f0) * (max(v_p, 0f0) - max(v_m, 0f0))
        conv!(V_temp_1, V_temp_2, p.k_gauss_2, p)
        conv!(A_temp, V_temp_1, p.k_C_A, p)
        conv!(B_temp, V_temp_1, p.k_C_B, p)
        @. B_temp = abs(B_temp)
        @. v_C = p.γ * (max(A_temp - B_temp, 0f0) + max(-A_temp - B_temp, 0f0))
    end
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
    @inbounds begin
    @. dx =
        p.δ_c * (
            -x + (
                (1.0f0 - x) * (
                    (p.α * C) + (p.ϕ * max(z - p.Γ, 0f0)) .+ (p.v_21 * x_v2) +
                    p.att
                )
            )
        )
    end
    return nothing
end



#     L4 excit
function fun_dy!(
    dy::AbstractArray,
    y::AbstractArray,
    C::AbstractArray,
    x::AbstractArray,
    m::AbstractArray,
    dy_temp::AbstractArray,
    p::NamedTuple,
)
    @inbounds begin
    conv!(dy_temp, m, p.k_W_p, p)
    @. dy_temp = m * dy_temp
    fun_f!(dy_temp, p)
    @. dy = p.δ_c * (-y + ((1f0 - y) * (C + (p.η_p * x))) - ((1f0 + y) * dy_temp))
    end
    return nothing
end


#     l4 inhib
function fun_dm!(
    dm::AbstractArray,
    m::AbstractArray,
    x::AbstractArray,
    dm_temp::AbstractArray,
    p::NamedTuple,
)
    @inbounds begin
    conv!(dm_temp, m, p.k_W_m, p)
    fun_f!(dm_temp, p)
    @. dm = p.δ_m * (-m + (p.η_m * x) - (m * dm_temp))
    end
    return nothing
end



#     L2/3 excit
function fun_dz!(
    dz::AbstractArray,
    z::AbstractArray,
    y::AbstractArray,
    H_z::AbstractArray,
    s::AbstractArray,
    dz_temp::AbstractArray,
    p::NamedTuple,
)
    @inbounds begin
    conv!(dz_temp, s, p.k_T_p, p)
    @. dz =
        p.δ_z * (
            -z + ((1f0 - z) * ((p.λ * max(y, 0f0)) + H_z + (p.a_23_ex * p.att))) -
            ((z + p.ψ) * dz_temp)
        )
    end
    return nothing
end

#     L2/3 inhib
function fun_ds!(
    ds::AbstractArray,
    s::AbstractArray,
    H_z::AbstractArray,
        ds_temp::AbstractArray,
    p::NamedTuple,
)
    @inbounds begin
    conv!(ds_temp, s, p.k_T_m, p)
    @. ds = p.δ_s * (-s + H_z + (p.a_23_in * p.att) - (s * ds_temp))
        end
    return nothing
end


function fun_H_z!(
        H_z::AbstractArray,
        z::AbstractArray,
        H_z_temp::AbstractArray,
        p::NamedTuple)
    @inbounds begin
    @. H_z_temp = max(z - p.Γ, 0f0)
    conv!(H_z, H_z_temp, p.k_H, p)
    end
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
    @inbounds begin
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
    end
    return nothing
end


# V2 L4 excit
function fun_dy_v2!(
    dy::AbstractArray,
    y_v2::AbstractArray,
    z::AbstractArray,
    x_v2::AbstractArray,
    m_v2::AbstractArray,
        dy_temp::AbstractArray,
    p::NamedTuple,
)
    @inbounds begin
    conv!(dy_temp, m_v2, p.k_W_p, p)
    fun_f!(dy_temp, p)
    @. dy =
        p.δ_c * (
            -y_v2 +
            ((1.0f0 - y_v2) * ((p.v12_4 * max(z - p.Γ, 0f0)) + (p.η_p * x_v2))) -
            ((1.0f0 + y_v2) * dy_temp)
        )
    end
    return nothing
end



# # # for equilabrium


# # l6 equilabrium
function fun_x_equ!(x::AbstractArray, C::AbstractArray, z::AbstractArray, x_v2::AbstractArray, x_temp::AbstractArray, p::NamedTuple)
    @inbounds begin
        @. x_temp = (p.α * C) + (p.ϕ * max(z - p.Γ, 0f0)) + (p.v_21 * x_v2) + p.att
        @. x = x_temp /(1+x_temp)
    end
    return nothing 
end


# # l4 excit equilabrium
function fun_y_equ!(y::AbstractArray, C::AbstractArray, x::AbstractArray, m::AbstractArray, dy_temp::AbstractArray, p::NamedTuple)
    @inbounds begin
        conv!(dy_temp, m, p.k_W_p, p)
        @. dy_temp = m * dy_temp
        fun_f!(dy_temp, p)
        @. y = (C + (p.η_p * x) - dy_temp)/(1 + C + (p.η_p * x) + dy_temp)
    end
    return nothing
end




end



