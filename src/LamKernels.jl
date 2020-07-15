"""
# module kernels

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-06-07

# Examples

```jldoctest
julia>
```
"""


module LamKernels

export kern_A, kern_B

# lgn to l6/l4
using NNlib, ImageFiltering, Images, OffsetArrays

function kern_d_ph(σ::Real, θ::Real, l = 4*ceil(Int,σ)+1)
    isodd(l) || throw(ArgumentError("length must be odd"))
    w = l>>1
    g = σ == 0 ? [1f0] : [exp(-x*cos(θ)/(2f0*σ)) for x=-w:w]
    centered(g/sum(g))
end

function kern_d_pv(σ::Real, θ::Real, l = 4*ceil(Int,σ)+1)
    isodd(l) || throw(ArgumentError("length must be odd"))
    w = l>>1
    g = σ == 0 ? [1f0] : [exp(-x*sin(θ)/(2f0*σ)) for x=-w:w]
    centered(g/sum(g))
end

function kern_d_mh(σ::Real, θ::Real, l = 4*ceil(Int,σ)+1)
    isodd(l) || throw(ArgumentError("length must be odd"))
    w = l>>1
    g = σ == 0 ? [1f0] : [exp(x*cos(θ)/(2f0*σ)) for x=-w:w]
    centered(g/sum(g))
end

function kern_d_mv(σ::Real, θ::Real, l = 4*ceil(Int,σ)+1)
    isodd(l) || throw(ArgumentError("length must be odd"))
    w = l>>1
    g = σ == 0 ? [1f0] : [exp(x*sin(θ)/(2f0*σ)) for x=-w:w]
    centered(g/sum(g))
end

function kern_d_p(σ::Real, θ::Real, l = 4*ceil(Int,σ)+1)
    kern_d_pv(σ, θ, l) .* transpose(kern_d_ph(σ, θ, l))
end

function kern_d_m(σ::Real, θ::Real, l = 4*ceil(Int,σ)+1)
    kern_d_mv(σ, θ, l) .* transpose(kern_d_mh(σ, θ, l))
end

function kern_A(σ::Real, θ::Real, l = 4*ceil(Int,σ)+1)
    kern_d_p(σ, θ, l).-kern_d_m(σ, θ, l)
end


# function kern_A(σ::Real, θ::Real, l = 4*ceil(Int,σ)+1)
#     relu.(kern_d(σ, θ, l)) .- relu.(-1 .*(kern_d(σ, θ, l)))
# end

function kern_B(σ::Real, θ::Real, l = 4*ceil(Int,σ)+1)
    relu.(kern_A(σ, θ, l)) .+ relu.(-1f0 .*(kern_A(σ, θ, l)))
end




function gaussian_rot(σ_x::Real, σ_y::Real, θ::Real, l = 4*ceil(Int, max(σ_a,σ_b))+1)
    isodd(l) || throw(ArgumentError("length must be odd"))
    w = l>>1
    g = OffsetArray(fill(0.0f0, l, l), -w:w, -w:w)
    #todo add when σ_x or/and σ_y == 0
    for x ∈ -w:w, y ∈ -w:w
        g[x,y] = exp(-1f0/2f0*((((x*cos(θ)-y*sin(θ))/σ_x)^2f0)+(((x*sin(θ)+y*cos(θ))/σ_y)^2f0)))
    end
    centered(g/sum(g))
end



# Rotating function
function fun_R(x,y,θ)
    x*cos(θ)-y*sin(θ), x*sin(θ)+y*cos(θ)
end
#
#
# function kern_H()
#     # todo:
# end
#
#
# function kern_H_v2()
#     # todo:
# end
#
#
# function W_p()
#     # todo:
# end
#
#
# function W_m()
#     # todo:
# end
#
#
# function T_m()
#     # todo:
#
# end
#
#
# function T_P()
#     # todo:
# end
#
# # T_P_11 = 0.9032
# # T_P_21 = 0.1384
# # T_P_12 = 0.1282
# # T_P_22 = 0.8443
# # T_M_11 = 0.2719
# # T_M_21 = 0.0428
# # T_M_12 = 0.0388
# # T_M_22 = 0.2506
#
# # T_P in V2 0.625x T_P in V1  ??
#
#


end