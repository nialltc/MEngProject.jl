# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl,md
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Julia 1.4.0
#     language: julia
#     name: julia-1.4
# ---

using PyPlot, NNlib,  ImageFiltering, Images, MEngProject, MEngProject.LamKernels, MEngProject.Laminart, MEngProject.Utils

σ_2 = 0.5
θ = pi/3
K = 12;

# + pycharm={"name": "#%%\n"}
for k=1:1:2K
    θ = π*(k-1)/K
    Utils.plot_rb(LamKernels.kern_d_pv(σ_2, θ))
end

# + pycharm={"name": "#%%\n"}
for k=1:1:2K
    θ = π*(k-1)/K
    Utils.plot_rb(LamKernels.kern_d_ph(σ_2, θ))
end   

# + pycharm={"name": "#%%\n"}
for k=1:1:2K
    θ = π*(k-1)/K
    Utils.plot_rb(LamKernels.kern_d_mv(σ_2, θ))
end   

# + pycharm={"name": "#%%\n"}
for k=1:1:2K
    θ = π*(k-1)/K
    Utils.plot_rb(LamKernels.kern_d_mh(σ_2, θ))
end 

# + pycharm={"name": "#%%\n"}
for k=1:1:2K
    θ = π*(k-1)/K
    Utils.plot_rb(LamKernels.kern_d_p(σ_2, θ))
end 

# + pycharm={"name": "#%%\n"}
 

# + pycharm={"name": "#%%\n"}
for k=1:1:2K
    θ = π*(k-1)/K
    Utils.plot_rb(LamKernels.kern_d_m(σ_2, θ))
end   

# + pycharm={"name": "#%%\n"}
for k=1:1:2K
    θ = π*(k-1)/K
    Utils.plot_rb(LamKernels.kern_d(σ_2, θ))
end   

# + pycharm={"name": "#%%\n"}
for k=1:1:2K
    θ = π*(k-1)/K
    Utils.plot_rb(LamKernels.kern_d(σ_2, θ))
end   

# + pycharm={"name": "#%%\n"}
for k=1:1:2K
    θ = π*(k-1)/K
    Utils.plot_rb(LamKernels.kern_b(σ_2, θ))
end   

# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}


# +
        
Gray.(-(LamKernels.kern_d(σ_2, θ)-LamKernels.kern_d_p(σ_2, θ)))

# + pycharm={"name": "#%%\n"}
@manipulate for k=1:1:2K
    θ = π*(k-1)/K
    Gray.(3 * LamKernels.kern_d(σ_2, θ))
end

# + pycharm={"name": "#%%\n"}
@manipulate for k=1:1:2K
    θ = π*(k-1)/K
    Gray.(3 * LamKernels.kern_b(σ_2, θ))
end

# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}

# -
















