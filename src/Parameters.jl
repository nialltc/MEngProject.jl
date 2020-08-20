"""
# module parameters

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-06-07

# Examples

```jldoctest
julia>
```
"""
module Parameters
p_temp_f32 = (σ_1 = 1.0f0,
σ_2 = 0.5f0,
H_σ_x = 3.0f0,
H_σ_y = 4.0f0)

parameters_f32 = (K = 2,
δ_v = 1.25f0,
δ_c = 0.25f0,
δ_m = 0.01875f0,
δ_z = 0.125f0,
δ_s = 2.5f0,
σ_1 = p_temp_f32.σ_1,
C_1 = 1.5f0,
C_2 = 0.075f0,
σ_2 = p_temp_f32.σ_2,
γ = 10.0f0,
α = 0.5f0,
ϕ = 2.0f0,
Γ = 0.2f0,
v_21 = 0.0f0, #1.0
μ = 2.0f0,
ν = 1.1f0,
n = 6.0f0,
att = 0.0f0,# p25
η_p = 2.1f0,
η_m = 1.5f0,
λ = 1.5f0,
a_23_ex = 3.0f0,
a_23_in = 0.5f0,
v12_6 = 1.0f0,
v12_4 = 5.0f0,
ψ = 0.5f0,
C_AB_l =  4*ceil(Int, p_temp_f32.σ_2)+1,
H_σ_x = p_temp_f32.H_σ_x,
H_σ_y = p_temp_f32.H_σ_y,
H_fact = 5.0f0,
# H_l =  4*ceil(Int, max(p_temp.H_σ_x, p_temp.H_σ_y))+1,
H_l = 19,
T_fact = [0.87f0,0.13f0],       #avg TP same orient, other orient
T_p_m = 0.302f0,    #avg TM/TP
T_v2_fact = 0.625f0,     #T in V2 = T*
# W, H sizes
H_size = 19,
W_size = 19,
# customn parameters for controling feedback, lgn equlibrum
lgn_equ_u = 1.0f0,
lgn_equ_A = 0.0f0,
lgn_equ_B = 0.0f0,
filling = "circular",
W_l = 19)


p_temp_f64 = (σ_1 = 1.0,
σ_2 = 0.5,
H_σ_x = 3.0,
H_σ_y = 4.0)

parameters_f64 = (K = 2,
δ_v = 1.25,
δ_c = 0.25,
δ_m = 0.01875,
δ_z = 0.125,
δ_s = 2.5,
σ_1 = p_temp_f64.σ_1,
C_1 = 1.5,
C_2 = 0.075,
σ_2 = p_temp_f64.σ_2,
γ = 10.0,
α = 0.5,
ϕ = 2.0,
Γ = 0.2,
v_21 = 0.0, #1.0
μ = 2.0,
ν = 1.1,
n = 6.0,
att = 0.0,# p25
η_p = 2.1,
η_m = 1.5,
λ = 1.5,
a_23_ex = 3.0,
a_23_in = 0.5,
v12_6 = 1.0,
v12_4 = 5.0,
ψ = 0.5,
C_AB_l =  4*ceil(Int, p_temp_f64.σ_2)+1,
H_σ_x = p_temp_f64.H_σ_x,
H_σ_y = p_temp_f64.H_σ_y,
H_fact = 5.0,
# H_l =  4*ceil(Int, max(p_temp.H_σ_x, p_temp.H_σ_y))+1,
H_l = 19,
T_fact = [0.87,0.13],       #avg TP same orient, other orient
T_p_m = 0.302,    #avg TM/TP
T_v2_fact = 0.625,     #T in V2 = T*
# W, H sizes
H_size = 19,
W_size = 19,
# customn parameters for controling feedback, lgn equlibrum
lgn_equ_u = 1.0,
lgn_equ_A = 0.0,
lgn_equ_B = 0.0,
filling = "circular",
W_l = 19)



end

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

# T_m ~ T_p /3.2 calculated..


#
# const δ_v = 1.25
# const δ_c = 0.25
# const δ_m = 0.01875
# const δ_z = 0.125
# const δ_s = 2.5
# const δ_c = 0.25
#
# const σ_1 = 1
#
# const C_1 = 1.5
# const C_2 = 0.075
#
#
# const σ_2 = 0.5
# const K = 2
# const γ = 10
#
# const α = 0.5
#
# const ϕ = 2.0
#
# const Γ = 0.2
#
# const v_21 = 0 #1.0
#
# const μ = 2
# const ν = 1.1
# const n = 6
#
# const att = 0
# # p25
#
# const η_p = 2.1
#
# const η_m = 1.5
#
# const λ = 1.5
#
# const a_23_ex = 3
# const a_23_in = 0.5
#
# const v12_6 = 1
# const v12_4 = 5
#
# const ψ = 0.5
