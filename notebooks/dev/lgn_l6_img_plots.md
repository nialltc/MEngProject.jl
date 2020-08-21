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
using PyPlot, NNlib,  ImageFiltering, Images, MEngProject, MEngProject.LamKernels, MEngProject.Laminart, MEngProject.Utils
```

```julia
# matplotlib.rcParams["text.usetex"] = true
```

```julia
img = convert(Array{Float64,2}, load("../input_img/Iine_100_100_gs.png"));
```

```julia
const σ_1 = 1
const σ_2 = 0.5
const γ = 10;
const K = 12;
```

```julia
u_p = Laminart.I_u(img)
u_m = - u_p
v_p = fun_equ.(u_p)
v_m = fun_equ.(u_m)
v = relu.(v_p)-relu.(v_m)
V = exp(-1/8) .* imfilter(v, Kernel.gaussian(σ_2), "circular")

R = reshape(Array{eltype(V)}(undef, size(V)[1], size(V)[2]*K),size(V)[1],size(V)[2],K)
L = copy(R)

for k in 1:K
    θ = π*(k-1)/K
    R[:,:,k] = imfilter(V, relu.(LamKernels.kern_d(σ_2, θ)), "circular")
    L[:,:,k] = -imfilter(V, relu.(-LamKernels.kern_d(σ_2, θ)), "circular")
end

S_a = R .+ L
S_b = -abs.(R .- L)
S = γ .* relu.(S_a .+ S_b)
C = Laminart.fun_v_C(v_p, v_m, σ_2, K);
```

```julia
Utils.plot_rb(img, "img", true, -1, 1, true)
```

```julia
Utils.plot_rb(u_p)
```

```julia
Utils.plot_rb(u_m)
```

```julia
Utils.plot_rb(v_p)
```

```julia
Utils.plot_rb(v_m,  "img", true, -2,2)
```

```julia
Utils.plot_rb(v)
```

```julia
Utils.plot_rb(V)
```

```julia
Utils.save_orientations_rb(R, "img")
```

```julia

Utils.save_orientations_rb(L, "img")

```

```julia

Utils.save_orientations_gs(S, "img",0,1.2,true)
```

```julia
Utils.save_orientations_gs(C, "img", 0,2)
```

```julia
R_ = reshape(Array{eltype(V)}(undef, size(V)[1], size(V)[2],2*K),size(V)[1],size(V)[2],2*K)
L_ = copy(R_)

for k in 1:2*K
    θ = π*(k-1)/K
    R_[:,:,k] = imfilter(V, relu.(LamKernels.kern_d(σ_2, θ)), "circular")
    L_[:,:,k] = -imfilter(V, relu.(-LamKernels.kern_d(σ_2, θ)), "circular")
end

S_a_ = R_ .+ L_
S_b_ = -abs.(R_ .- L_)
S_ = γ .* relu.((S_a_ .+ S_b_));
```

```julia
C_ = reshape(Array{eltype(V)}(undef, size(V)[1], size(V)[2],K),size(V)[1],size(V)[2],K)

for k in 1:K
    C_[:,:,k] = S_[:,:,k] .+ S_[:,:,k+K]  
end
```

```julia
Utils.save_orientations_gs(C_, "img", 0, 2.2)
```

```julia
findmax(abs.(C-C_))
```

```julia

```