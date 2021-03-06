---
jupyter:
  jupytext:
    formats: ipynb,jl,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Julia 1.4.0
    language: julia
    name: julia-1.4
---

```julia
using NNlib,  Interact, ImageFiltering, Images, TestImages, ImageView, Plots, OffsetArrays
```

```julia
x=-2:.2:2
y=-2:.2:2

σ_2 = 0.5
δ = σ_2/2
K = 12;
```

```julia
G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
Plots.wireframe(x,y,G)
```

```julia
@manipulate for k=1:1:2K
    θ = π*(k-1)/K
    G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
    D(x,y) = G(x+(δ*cos(θ)), y + (δ*sin(θ))) - G(x - (δ*cos(θ)), y - (δ*sin(θ)))  
    Plots.wireframe(x,y,D)
end
```

```julia
@manipulate for k=1:1:2K
    θ = π*(k-1)/K
    G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
    D(x,y) = G(x+(δ*cos(θ)), y + (δ*sin(θ))) - G(x - (δ*cos(θ)), y - (δ*sin(θ)))
    R(x,y) = relu.(D(x,y))
#     L(x,y) = -relu.(-D(x,y))
    Plots.wireframe(x,y,R)
end
```

```julia
@manipulate for k=1:1:2K
    θ = π*(k-1)/K
    G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
    D(x,y) = G(x+(δ*cos(θ)), y + (δ*sin(θ))) - G(x - (δ*cos(θ)), y - (δ*sin(θ)))
#     R(x,y) = relu.(D(x,y))
    L(x,y) = -relu.(-D(x,y))
    Plots.wireframe(x,y,L)
end
```

```julia
@manipulate for k=1:1:2K
    θ = π*(k-1)/K
    G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
    D(x,y) = G(x+(δ*cos(θ)), y + (δ*sin(θ))) - G(x - (δ*cos(θ)), y - (δ*sin(θ)))
    R(x,y) = relu.(D(x,y))
    L(x,y) = -relu.(-D(x,y))
    S_a(x,y) = R(x,y) + L(x,y)
    Plots.wireframe(x,y,S_a)
end
```

```julia
@manipulate for k=1:1:2K
    θ = π*(k-1)/K
    G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
    D(x,y) = G(x+(δ*cos(θ)), y + (δ*sin(θ))) - G(x - (δ*cos(θ)), y - (δ*sin(θ)))
    R(x,y) = relu.(D(x,y))
    L(x,y) = -relu.(-D(x,y))
    S_b(x,y) = R(x,y) - L(x,y)
    Plots.wireframe(x,y,S_b)
end
```

```julia

```

```julia

```

```julia
@manipulate for k=1:1:2K, σ_2=0.2:0.05:5
    θ = π*(k-1)/K
#     G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
    d(x,y) = exp(-(x*cos(θ)+y*sin(θ))/2σ_2)-exp((x*cos(θ)+y*sin(θ))/2σ_2)
#     D_sep(x,y) = G(x,y)*exp(-1/8)*d(x,y)
#     d_sep_rp(x,y) = relu.(d(x,y))
#     d_sep_rm(x,y) = relu.(-d(x,y))
#     A(x,y) = d_sep_rp(x,y) - d_sep_rm(x,y)
#     B(x,y) = d_sep_rp(x,y) + d_sep_rm(x,y)
    Plots.wireframe(x,y,d)
end
```

```julia
@manipulate for k=1:1:2K
    θ = π*(k-1)/K
    G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
    d(x,y) = exp(-(x*cos(θ)+y*sin(θ))/2σ_2)-exp((x*cos(θ)+y*sin(θ))/2σ_2)
    D_sep(x,y) = G(x,y)*exp(-1/8)*d(x,y)
#     d_sep_rp(x,y) = relu.(d(x,y))
#     d_sep_rm(x,y) = relu.(-d(x,y))
#     A(x,y) = d_sep_rp(x,y) - d_sep_rm(x,y)
#     B(x,y) = d_sep_rp(x,y) + d_sep_rm(x,y)
    Plots.wireframe(x,y,D_sep)
end
```

```julia
@manipulate for k=1:1:2K
    θ = π*(k-1)/K
    G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
    D(x,y) = G(x+(δ*cos(θ)), y + (δ*sin(θ))) - G(x - (δ*cos(θ)), y - (δ*sin(θ)))  
    Plots.wireframe(x,y,D)
end
```

```julia
@manipulate for k=1:1:2K
    θ = π*(k-1)/K
#     G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
    d(x,y) = exp(-(x*cos(θ)+y*sin(θ))/2σ_2)-exp((x*cos(θ)+y*sin(θ))/2σ_2)
#     D_sep(x,y) = G(x,y)*exp(-1/8)*d(x,y)
    d_sep_rp(x,y) = relu.(d(x,y))
#     d_sep_rm(x,y) = relu.(-d(x,y))
#     A(x,y) = d_sep_rp(x,y) - d_sep_rm(x,y)
#     B(x,y) = d_sep_rp(x,y) + d_sep_rm(x,y)
    Plots.wireframe(x,y,d_sep_rp)
end
```

```julia
@manipulate for k=1:1:2K
    θ = π*(k-1)/K
#     G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
    d(x,y) = exp(-(x*cos(θ)+y*sin(θ))/2σ_2)-exp((x*cos(θ)+y*sin(θ))/2σ_2)
#     D_sep(x,y) = G(x,y)*exp(-1/8)*d(x,y)
#     d_sep_rp(x,y) = relu.(d(x,y))
    d_sep_rm(x,y) = relu.(-d(x,y))
#     A(x,y) = d_sep_rp(x,y) - d_sep_rm(x,y)
#     B(x,y) = d_sep_rp(x,y) + d_sep_rm(x,y)
    Plots.wireframe(x,y,d_sep_rm)
end
```

```julia
@manipulate for k=1:1:2K
    θ = π*(k-1)/K
    δ = σ_2/2
#     G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
    d(x,y) = exp(-(x*cos(θ)+y*sin(θ))/2σ_2)-exp((x*cos(θ)+y*sin(θ))/2σ_2)
#     D_sep(x,y) = G(x,y)*exp(-1/8)*d(x,y)
    d_sep_rp(x,y) = relu.(d(x,y))
    d_sep_rm(x,y) = relu.(-d(x,y))
    A(x,y) = d_sep_rp(x,y) - d_sep_rm(x,y)
#     B(x,y) = d_sep_rp(x,y) + d_sep_rm(x,y)
    Plots.wireframe(x,y,A)
end
```

```julia
@manipulate for k=1:1:2K
    θ = π*(k-1)/K
#     G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
    d(x,y) = exp(-(x*cos(θ)+y*sin(θ))/2σ_2)-exp((x*cos(θ)+y*sin(θ))/2σ_2)
#     D_sep(x,y) = G(x,y)*exp(-1/8)*d(x,y)
    d_sep_rp(x,y) = relu.(d(x,y))
    d_sep_rm(x,y) = relu.(-d(x,y))
    A(x,y) = d_sep_rp(x,y) - d_sep_rm(x,y)
    B(x,y) = d_sep_rp(x,y) + d_sep_rm(x,y)
    Plots.wireframe(x,y,B)
end
```

```julia
x=-10:1:10
y=-10:1:10
k=1
θ = π*(k-1)/K
#     G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
d(x,y) = exp(-(x*cos(θ)+y*sin(θ))/2σ_2)-exp((x*cos(θ)+y*sin(θ))/2σ_2)
#     D_sep(x,y) = G(x,y)*exp(-1/8)*d(x,y)
d_sep_rp(x,y) = relu.(d(x,y))
d_sep_rm(x,y) = relu.(-d(x,y))
A(x,y) = d_sep_rp(x,y) - d_sep_rm(x,y)
B(x,y) = d_sep_rp(x,y) + d_sep_rm(x,y)
Plots.wireframe(x,y,A)

```

```julia
x=-10:1:10
y=-10:1:10
k=1
θ = π*(k-1)/K
#     G(x,y) = 1/(2*π*σ_2^2)*exp(-1/(2*σ_2^2)*(x^2+y^2))
d(x,y) = exp(-(x*cos(θ)+y*sin(θ))/2σ_2)-exp((x*cos(θ)+y*sin(θ))/2σ_2)
#     D_sep(x,y) = G(x,y)*exp(-1/8)*d(x,y)
d_sep_rp(x,y) = relu.(d(x,y))
d_sep_rm(x,y) = relu.(-d(x,y))
A(x,y) = d_sep_rp(x,y) - d_sep_rm(x,y)
B(x,y) = d_sep_rp(x,y) + d_sep_rm(x,y)
Plots.wireframe(x,y,B)
```

```julia

```

```julia

```

```julia

```
