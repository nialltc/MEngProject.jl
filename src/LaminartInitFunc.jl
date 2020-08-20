"""
# module LaminartParameterFunc

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-08-20

# Examples

```jldoctest
julia>
```
"""
module LaminartInitFunc

import("./LaminartEqConv.jl")
import("./LaminartEqImfilter.jl")

using NNlib, ImageFiltering, Images, OffsetArrays, CUDA

function paraameterInit_conv_gpu(imgLoc::String, p::NamedTuple)
    img = convert(Array{Float32,2},  load(imgLoc));
    img = reshape2d_4d(img)
    img = cu(img)

    r = similar(img)

    parameters = kernels_gpu(img, p);

    LaminartEqConv.I_u!(r, img, parameters)
    temp_out = (I = img, r = r)
    parameters = merge(parameters, temp_out)
    return parameters
end


function paraameterInit_conv_cpu(imgLoc::String, p::NamedTuple)
    img = convert(Array{Float32,2},  load(imgLoc));
    img = reshape2d_4d(img)

    r = similar(img)

    parameters = kernels_cpu(img, p);

    LaminartEqConv.I_u!(r, img, parameters)
    temp_out = (I = img, r = r)
    parameters = merge(parameters, temp_out)
    return parameters
end


function paraameterInit_imfil_cpu(imgLoc::String, p::NamedTuple)
    img = convert(Array{Float32,2}, load(imgLoc));

    r = similar(img)

    parameters = kernels_imfil_cpu(img, p);

    LaminartEqImfilter.I_u!(r, img, parameters)
    temp_out = (I = img, r = r)
    parameters = merge(parameters, temp_out)
    return parameters
end


function reshape2d_4d(img::AbstractArray)
    reshape(img, size(img)[1], size(img)[2], 1, 1)
end


function add_I_u_p(I::AbstractArray, p::NamedTuple)
# 	todo fix
# 	I_4d = cu(reshape2d_4d(I))
    I_4d = reshape2d_4d(I)
    r = similar(I_4d)
    I_u!(r, I_4d, p)
    temp_out = (I = I_4d, r = r)
    return merge(p, temp_out)
end


function kernels_conv_gpu(img::AbstractArray, p::NamedTuple)
       C_A_temp = reshape(
        Array{eltype(img)}(undef, p.C_AB_l, p.C_AB_l * p.K),
        p.C_AB_l,
        p.C_AB_l,
        1,
        p.K
    )
C_B_temp = similar(C_A_temp)
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
        C_A_temp[:, :, 1,k] = LamKernels.kern_A(p.σ_2, θ)
        C_B_temp[:, :, 1,k] = LamKernels.kern_B(p.σ_2, θ)
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

temp_out = (
        k_gauss_1 = cu(reshape2d_4d(Kernel.gaussian(p.σ_1))),
        k_gauss_2 = cu(reshape2d_4d(Kernel.gaussian(p.σ_2))),
        k_C_A = cu(C_A_temp),
        k_C_B = cu(C_B_temp),

# 		todo use mean of x_lgn?
        k_x_lgn = cu(reshape(ones(Float32,1,p.K),1,1,p.K,1)),
# 		k_x_lgn = cu(reshape(ones(Float32,1,p.K),1,1,p.K,1) ./ p.K),
        k_W_p = cu(W_temp),
        k_W_m = cu(W_temp),
        k_H = cu(H_temp),
        k_T_p = cu(T_temp),
        k_T_m = cu(p.T_p_m .* T_temp),
        k_T_p_v2 = cu(p.T_v2_fact .* T_temp),
        k_T_m_v2 = cu(p.T_v2_fact .* p.T_p_m .* T_temp),
        dim_i = size(img)[1],
        dim_j = size(img)[2],
        x_V2 = cu(reshape(zeros(Float32, size(img)[1], size(img)[2] * p.K), size(img)[1], size(img)[2],p.K,1)),
ν_pw_n= p.ν^p.n, )

merge(p, temp_out)
end

function kernels_conv_cpu(img::AbstractArray, p::NamedTuple)
       C_A_temp = reshape(
        Array{eltype(img)}(undef, p.C_AB_l, p.C_AB_l * p.K),
        p.C_AB_l,
        p.C_AB_l,
        1,
        p.K
    )
C_B_temp = similar(C_A_temp)
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
        C_A_temp[:, :, 1,k] = LamKernels.kern_A(p.σ_2, θ)
        C_B_temp[:, :, 1,k] = LamKernels.kern_B(p.σ_2, θ)
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


    temp_out = (
        k_gauss_1 = reshape2d_4d(Kernel.gaussian(p.σ_1)),
        k_gauss_2 = reshape2d_4d(Kernel.gaussian(p.σ_2)),
        k_C_A = C_A_temp,
        k_C_B = C_B_temp,

# 		todo use mean of x_lgn?
        k_x_lgn = reshape(ones(Float32,1,p.K),1,1,p.K,1),
# 		k_x_lgn = reshape(ones(Float32,1,p.K),1,1,p.K,1)./p.K,
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


function kernels_imfil_cpu(img::AbstractArray, p::NamedTuple)
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
        C_A_temp[:, :, k] = reflect(centered(LamKernels.kern_A(p.σ_2, θ)))           #ij ijk ijk
        C_B_temp[:, :, k] = reflect(centered(LamKernels.kern_B(p.σ_2, θ)))               #ij ijk ijk
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



end