"""
# module kernels

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-06-07

functions to create LAMINART kernels for lgn -> L6, L4 and W kernel
# Examples

```jldoctest
julia>
```
"""


module LaminartKernels

export kern_A, kern_B

# lgn to l6/l4
using NNlib, ImageFiltering, Images, OffsetArrays

function kern_d_ph(σ::Real, θ::Real, l = 4 * ceil(Int, σ) + 1)
    isodd(l) || throw(ArgumentError("length must be odd"))
    w = l >> 1
    g = σ == 0 ? [1f0] : [exp(-x * cos(θ) / (2f0 * σ)) for x = -w:w]
    centered(g / sum(g))
end

function kern_d_pv(σ::Real, θ::Real, l = 4 * ceil(Int, σ) + 1)
    isodd(l) || throw(ArgumentError("length must be odd"))
    w = l >> 1
    g = σ == 0 ? [1f0] : [exp(-x * sin(θ) / (2f0 * σ)) for x = -w:w]
    centered(g / sum(g))
end

function kern_d_mh(σ::Real, θ::Real, l = 4 * ceil(Int, σ) + 1)
    isodd(l) || throw(ArgumentError("length must be odd"))
    w = l >> 1
    g = σ == 0 ? [1f0] : [exp(x * cos(θ) / (2f0 * σ)) for x = -w:w]
    centered(g / sum(g))
end

function kern_d_mv(σ::Real, θ::Real, l = 4 * ceil(Int, σ) + 1)
    isodd(l) || throw(ArgumentError("length must be odd"))
    w = l >> 1
    g = σ == 0 ? [1f0] : [exp(x * sin(θ) / (2f0 * σ)) for x = -w:w]
    centered(g / sum(g))
end

function kern_d_p(σ::Real, θ::Real, l = 4 * ceil(Int, σ) + 1)
    kern_d_pv(σ, θ, l) .* transpose(kern_d_ph(σ, θ, l))
end

function kern_d_m(σ::Real, θ::Real, l = 4 * ceil(Int, σ) + 1)
    kern_d_mv(σ, θ, l) .* transpose(kern_d_mh(σ, θ, l))
end

function kern_A(σ::Real, θ::Real, l = 4 * ceil(Int, σ) + 1)
    kern_d_p(σ, θ, l) .- kern_d_m(σ, θ, l)
end


# function kern_A(σ::Real, θ::Real, l = 4*ceil(Int,σ)+1)
#     relu.(kern_d(σ, θ, l)) .- relu.(-1 .*(kern_d(σ, θ, l)))
# end

function kern_B(σ::Real, θ::Real, l = 4 * ceil(Int, σ) + 1)
    relu.(kern_A(σ, θ, l)) .+ relu.(-1f0 .* (kern_A(σ, θ, l)))
end



"""
used to make W kernels
returns 2D kernel with elongated Gaussian rotated
"""
function gaussian_rot(
    σ_x::Real,
    σ_y::Real,
    θ::Real,
    l = 4 * ceil(Int, max(σ_a, σ_b)) + 1,
)
    isodd(l) || throw(ArgumentError("length must be odd"))
    w = l >> 1
    g = OffsetArray(fill(0.0f0, l, l), -w:w, -w:w)
    #todo add when σ_x or/and σ_y == 0
    for x ∈ -w:w, y ∈ -w:w
        g[x, y] = exp(
            -1f0 / 2f0 * (
                (((x * cos(θ) - y * sin(θ)) / σ_x)^2f0) +
                (((x * sin(θ) + y * cos(θ)) / σ_y)^2f0)
            ),
        )
    end
    centered(g / sum(g))
end



# Rotating function
function fun_R(x, y, θ)
    x * cos(θ) - y * sin(θ), x * sin(θ) + y * cos(θ)
end



end
