#=
kernel_plots:
- Julia version: 1.4.0
- Author: niallcullinane
- Date: 2020-06-07
=#

using MEngProject
using NNlib, ImageFiltering, Images, ImageView, Plots, OffsetArrays, MEngProject


cd("/Users/niallcullinane/Dropbox/GCIPA/Project/TestImages/")
# img = convert(Array{Float64,2}, load("single_line.png"));
img = convert(Array{Float64,2}, load("I_100_100_gs.png"));
img = convert(Array{Float64,2}, load("Iine_100_100_gs.png"))



l = [img, u, -u, img_90, u_rot, -u_rot]

save_2d_list(l, "u")


σ_2, θ_2=√.5
K = 8

u = I_u(img)



@time C = fun_v_C(u, -u, σ_2, θ_2, K)

save_orientations(C, "C")



img_90 = imrotate(img, π/2, axes(img))

u_rot = I_u(img_90)

C_rot = fun_v_C(u_rot, -u_rot, σ_2, θ, K)

save_orientations(C_rot, "C_rot90")



# kernel plots
# each at 2 orientations
kern_d_pv(σ_2, θ)
kern_d_ph(σ_2, θ)
kern_d_mv(σ_2, θ)
kern_d_mh(σ_2, θ)
kern_d_p(σ_2, θ)
kern_d_m(σ_2, θ)
kern_d(σ_2, θ)
kern_A(σ_2, θ)
kern_B(σ_2, θ)

# inter image plots
imgA
imgB
relu.(A .- B)
relu.(-A .- B)
