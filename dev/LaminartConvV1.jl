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
module LaminartConv
include("./LamKernels.jl")

using NNlib, ImageFiltering, Images, OffsetArrays, CUDA
# , MEngProject.LamKernels

export I_u, fun_v_C, fun_equ

struct MyFunction{T} <: Function
  x_lgn::T
  C::T
  H_z::T
end

# function (ff::MyFunction)(du, u, p, t)
# # function f!(du, u, p, t)
#     @inbounds begin
#         x = CuArray(@view u[:, :, 1:p.K,:])
#         y = CuArray(@view u[:, :, p.K+1:2*p.K,:])
#         m = CuArray(@view u[:, :, 2*p.K+1:3*p.K,:])
#         z = CuArray(@view u[:, :, 3*p.K+1:4*p.K,:])
#         s = CuArray(@view u[:, :, 4*p.K+1:5*p.K,:])

#         v_p = CuArray(@view u[:, :, 5*p.K+1:5*p.K+1,:])
#         v_m = CuArray(@view u[:, :, 5*p.K+2:5*p.K+2,:])

#         dx = CuArray(@view du[:, :, 1:p.K,:])
#         dy = CuArray(@view du[:, :, p.K+1:2*p.K,:])
#         dm = CuArray(@view du[:, :, 2*p.K+1:3*p.K,:])
#         dz = CuArray(@view du[:, :, 3*p.K+1:4*p.K,:])
#         ds = CuArray(@view du[:, :, 4*p.K+1:5*p.K,:])

#         dv_p = CuArray(@view du[:, :, 5*p.K+1:5*p.K+1,:])
#         dv_m = CuArray(@view du[:, :, 5*p.K+2:5*p.K+2,:])

#         fun_x_lgn!(ff.x_lgn, x, p)
#         fun_v_C!(ff.C, v_p, v_m, p)
#         fun_H_z!(ff.H_z, z, p)

#         fun_dv!(dv_p, v_p, p.r, ff.x_lgn, p)
#         fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, p)
#         fun_dx_v1!(dx, x, ff.C, z, p.x_V2, p)
#         fun_dy!(dy, y, ff.C, x, m, p)
#         fun_dm!(dm, m, x, p)
#         fun_dz!(dz, z, y, ff.H_z, s, p)
#         fun_ds!(ds, s, ff.H_z, p)

#     end
#     return nothing
# end



function (ff::MyFunction)(du, u, p, t)
# function f!(du, u, p, t)
#     @inbounds begin
        x = @view u[:, :, 1:p.K,:]
        y = @view u[:, :, p.K+1:2*p.K,:]
        m = @view u[:, :, 2*p.K+1:3*p.K,:]
        z = @view u[:, :, 3*p.K+1:4*p.K,:]
        s = @view u[:, :, 4*p.K+1:5*p.K,:]

        v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
        v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

        dx = @view du[:, :, 1:p.K,:]
        dy = @view du[:, :, p.K+1:2*p.K,:]
        dm = @view du[:, :, 2*p.K+1:3*p.K,:]
        dz = @view du[:, :, 3*p.K+1:4*p.K,:]
        ds = @view du[:, :, 4*p.K+1:5*p.K,:]

        dv_p = @view du[:, :, 5*p.K+1:5*p.K+1,:]
        dv_m = @view du[:, :, 5*p.K+2:5*p.K+2,:]

        fun_x_lgn!(ff.x_lgn, x, p)
        fun_v_C!(ff.C, v_p, v_m, p)
        fun_H_z!(ff.H_z, z, p)

        fun_dv!(dv_p, v_p, p.r, ff.x_lgn, p)
        fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, p)
        fun_dx_v1!(dx, x, ff.C, z, p.x_V2, p)
#         fun_dy!(dy, y, ff.C, x, m, p)
#         fun_dm!(dm, m, x, p)
#         fun_dz!(dz, z, y, ff.H_z, s, p)
#         fun_ds!(ds, s, ff.H_z, p)

#     end
    return nothing
end



# function (ff::MyFunction)(du, u, p, t)
# # function f!(du, u, p, t)
#     @inbounds begin
#         x = @view u[:, :, 1:p.K,:]
#         y = @view u[:, :, p.K+1:2*p.K,:]
#         m = @view u[:, :, 2*p.K+1:3*p.K,:]
#         z = @view u[:, :, 3*p.K+1:4*p.K,:]
#         s = @view u[:, :, 4*p.K+1:5*p.K,:]

#         v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
#         v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

#         dx = @view du[:, :, 1:p.K,:]
#         dy = @view du[:, :, p.K+1:2*p.K,:]
#         dm = @view du[:, :, 2*p.K+1:3*p.K,:]
#         dz = @view du[:, :, 3*p.K+1:4*p.K,:]
#         ds = @view du[:, :, 4*p.K+1:5*p.K,:]

#         dv_p = @view du[:, :, 5*p.K+1:5*p.K+1,:]
#         dv_m = @view du[:, :, 5*p.K+2:5*p.K+2,:]

#         fun_x_lgn!(ff.x_lgn[:,:,:,:], x[:,:,:,:], p)
#         fun_v_C!(ff.C[:,:,:,:], v_p[:,:,:,:], v_m[:,:,:,:], p)
#         fun_H_z!(ff.H_z[:,:,:,:], z[:,:,:,:], p)

#         fun_dv!(dv_p[:,:,:,:], v_p[:,:,:,:], p.r[:,:,:,:], ff.x_lgn[:,:,:,:], p)
#         fun_dv!(dv_m[:,:,:,:], v_m[:,:,:,:], .-p.r[:,:,:,:], ff.x_lgn[:,:,:,:], p)
#         fun_dx_v1!(dx[:,:,:,:], x[:,:,:,:], ff.C[:,:,:,:], z[:,:,:,:], p.x_V2[:,:,:,:], p)
#         fun_dy!(dy[:,:,:,:], y[:,:,:,:], ff.C[:,:,:,:], x[:,:,:,:], m[:,:,:,:], p)
#         fun_dm!(dm[:,:,:,:], m[:,:,:,:], x[:,:,:,:], p)
#         fun_dz!(dz[:,:,:,:], z[:,:,:,:], y[:,:,:,:], ff.H_z[:,:,:,:], s[:,:,:,:], p)
#         fun_ds!(ds[:,:,:,:], s[:,:,:,:], ff.H_z[:,:,:,:], p)

#     end
#     return nothing
# end



function kernels(img::AbstractArray, p::NamedTuple)
       C_Q_temp = reshape(
        Array{eltype(img)}(undef, p.C_AB_l, p.C_AB_l * p.K),
        p.C_AB_l,
        p.C_AB_l,
        1,
		p.K
    )
C_P_temp = similar(C_Q_temp)
	    H_temp = reshape(
        zeros(eltype(img), p.H_l, p.H_l * p.K * p.K),
        p.H_l,
        p.H_l,
        p.K,
    p.K)
 T_temp = reshape(Array{eltype(img)}(undef, p.K * p.K), 1, 1, p.K, p.K)     
 W_temp =
        reshape(Array{eltype(img)}(undef, p.W_l, p.W_l * p.K * p.K), p.W_l, p.W_l, p.K, p.K)
    for k ∈ 1:p.K
        θ = π * (k - 1.0f0) / p.K
        C_Q_temp[:, :, 1,k] = LamKernels.kern_d(p.σ_2, θ)          
        C_P_temp[:, :, 1,k] = LamKernels.kern_b(p.σ_2, θ)               
        H_temp[:, :, k,k] = p.H_fact .* LamKernels.gaussian_rot(p.H_σ_x, p.H_σ_y, θ, p.H_l)  
# 		todo make T kernel more general for higher K
        T_temp[1, 1, k,1] = p.T_fact[k]
        T_temp[1, 1, 2,2] = p.T_fact[1]
        T_temp[1, 1, 1,2] = p.T_fact[2]
        #todo: generalise T and W for higher K
        #         T_temp[:,:,k] = KernelFactors.gaussian(p.T_σ, p.K)
        #         for l ∈ 1:p.K
        #             W_temp[:,:,l,k] =
        #         end
    end

    W_temp[:, :, 1, 1] =
        5f0 .* LamKernels.gaussian_rot(3f0, 0.8f0, 0f0, p.W_l) .+
        LamKernels.gaussian_rot(0.4f0, 1f0, 0f0, p.W_l)
    
	W_temp[:, :, 2, 2] =
        5f0 .* LamKernels.gaussian_rot(3f0, 0.8f0,  π / 2f0, p.W_l) .+
        LamKernels.gaussian_rot(0.4f0, 1f0, π / 2f0, p.W_l)
    
	W_temp[:, :, 1, 2] = relu.(
        0.2f0 .- LamKernels.gaussian_rot(2f0, 0.6f0, 0f0, p.W_l) .-
        LamKernels.gaussian_rot(0.3f0, 1.2f0, 0f0, p.W_l))
    
	W_temp[:, :, 2, 1] = relu.(
        0.2f0 .- LamKernels.gaussian_rot(2f0, 0.6f0, π / 2f0, p.W_l) .-
        LamKernels.gaussian_rot(0.3f0, 1.2f0, π / 2f0, p.W_l))
	
#     W_temp[:, :, 1, 1] =
#         5f0 .* LamKernels.gaussian_rot(3f0, 0.8f0, 0f0, p.W_l) .+
#         LamKernels.gaussian_rot(0.4f0, 1f0, 0f0, p.W_l)
#     W_temp[:, :, 2, 2] =
#         5f0 .* LamKernels.gaussian_rot(3f0, 0.8f0, 0f0, p.W_l) .+
#         LamKernels.gaussian_rot(0.4f0, 1f0, π / 2f0, p.W_l)
#     W_temp[:, :, 1, 2] = relu.(
#         0.2f0 .- LamKernels.gaussian_rot(2f0, 0.6f0, 0f0, p.W_l) .-
#         LamKernels.gaussian_rot(0.3f0, 1.2f0, 0f0, p.W_l))
#     W_temp[:, :, 2, 1] = relu.(
#         0.2f0 .- LamKernels.gaussian_rot(2f0, 0.6f0, 0f0, p.W_l) .-
#         LamKernels.gaussian_rot(0.3f0, 1.2f0, π / 2f0, p.W_l))
	
temp_out = (
        k_gauss_1 = reshape2d_4d(Kernel.gaussian(p.σ_1)),
        k_gauss_2 = reshape2d_4d(Kernel.gaussian(p.σ_2)),
        k_C_d = C_Q_temp,
        k_C_b = C_P_temp,
		
# 		todo use mean of x_lgn?
		k_x_lgn = reshape(ones(Float32,1,p.K),1,1,p.K,1),
# 		k_x_lgn = reshape(ones(Float32,1,p.K),1,1,p.K,1))./p.K,
        k_W_p = W_temp,
        k_W_m = W_temp,
        k_H = H_temp,
        k_T_p = T_temp,
        k_T_m = p.T_p_m .* T_temp,
        k_T_p_v2 = p.T_v2_fact .* T_temp,
        k_T_m_v2 = p.T_v2_fact .* p.T_p_m .* T_temp,
        dim_i = size(img)[1],
        dim_j = size(img)[2],
        x_V2 = reshape(zeros(Float32, size(img)[1], size(img)[2] * p.K), size(img)[1], size(img)[2],p.K,1),
ν_pw_n= p.ν^p.n, )
merge(p, temp_out)
end


function reshape2d_4d(img::AbstractArray)
    reshape(img, size(img)[1], size(img)[2], 1, 1)
end

# function reshape_ijk_ij1k(img::AbstractArray, p::NamedTuple)
#     reshape(img, size(img)[1], size(img)[2], 1, p.K)
# end
	
	
function conv!(out::AbstractArray, img::AbstractArray, kern::AbstractArray, p::NamedTuple)
	out_ = @view out[:,:,:,:]
    out_ .= NNlib.conv(img, kern, pad=(size(kern)[1]>>1, size(kern)[1]>>1, size(kern)[2]>>1, size(kern)[2]>>1), flipped=true)
# 	@. out_ = out
    return nothing
end


function conv_dw!(out::AbstractArray, img::AbstractArray, kern::AbstractArray, p::NamedTuple)
     out = NNlib.depthwiseconv(img, kern, pad=(size(kern)[1]>>1, size(kern)[2]>>1, size(kern)[3]>>1, size(kern)[4]>>1), flipped=true)
    return nothing
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
# 	todo fix
	I_4d = CuArray(reshape2d_4d(I))
	r = similar(I_4d)
	I_u!(r, I_4d, p)
    temp_out = (I = I_4d, r = r)
    return merge(p, temp_out)
end


# retina

function I_u!(r::AbstractArray, I::AbstractArray, p::NamedTuple)
    
	conv!(r, I, p.k_gauss_1, p)
    @. r = I - r
end



# lgn feedback


function fun_x_lgn!(x_lgn::AbstractArray, x::AbstractArray, p::NamedTuple)
	out_ = @view x_lgn[:,:,:,:]
	    out_ .= NNlib.conv(x[:,:,:,:], p.k_x_lgn, pad=0,flipped=true)
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
    @. f_out = (p.μ * x^p.n) / (p.ν_pw_n + x^p.n)
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
	conv!(dv, x_lgn, p.k_gauss_1, p)
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
    conv!(V, temp, p.k_gauss_2, p)

    A = similar(v_C)
    #     allocate B to v_C
    
	conv!(A, V, p.k_C_d, p)
	conv!(v_C, V, p.k_C_b, p)
 
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
    conv!(dy, m, p.k_W_p, p)
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
    conv!(dm, m, p.k_W_m, p)
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
    conv!(dz, s, p.k_T_p, p)
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
    conv!(ds, s, p.k_T_m, p)
    @. ds = p.δ_s * (-s + H_z + (p.a_23_in * p.att) - (s * ds))
    return nothing
end


function fun_H_z!(H_z::AbstractArray, z::AbstractArray, p::NamedTuple)
	conv!(H_z, max.(z .- p.Γ, 0f0), p.k_H, p)
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
    conv!(dy, m_v2, p.k_W_p, p)
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

