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

module LaminartGPU

include("./LamKernels.jl")

using NNlib, ImageFiltering, Images, OffsetArrays, CUDA
# , MEngProject.LamKernels

export I_u, fun_v_C, fun_equ

mutable struct LamFunction{T <: AbstractArray} <: Function
	x::T
	m::T
	s::T

	x_lgn::T
	C::T
	H_z::T

	dy_temp::T
	dm_temp::T
	dz_temp::T
	ds_temp::T
	dv_temp::T
	H_z_temp::T
	V_temp_1::T
	V_temp_2::T
	Q_temp::T
	P_temp::T
end


function (ff::LamFunction)(du, u, p, t)

       @inbounds begin

		@. ff.x = @view u[:, :, 1:p.K,:]
        y = @view u[:, :, p.K+1:2*p.K,:]
        @. ff.m = @view u[:, :, 2*p.K+1:3*p.K,:]
        z = @view u[:, :, 3*p.K+1:4*p.K,:]
		@. ff.s = @view u[:, :, 4*p.K+1:5*p.K,:]

		v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
		v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

		dx = @view du[:, :, 1:p.K,:]
        dy = @view du[:, :, p.K+1:2*p.K,:]
        dm = @view du[:, :, 2*p.K+1:3*p.K,:]
        dz = @view du[:, :, 3*p.K+1:4*p.K,:]
        ds = @view du[:, :, 4*p.K+1:5*p.K,:]

		dv_p = @view du[:, :, 5*p.K+1:5*p.K+1,:]
		dv_m = @view du[:, :, 5*p.K+2:5*p.K+2,:]

		fun_x_lgn!(ff.x_lgn, ff.x, p)
        fun_v_C!(ff.C, v_p, v_m, ff.V_temp_1, ff.V_temp_2, ff.Q_temp, ff.P_temp, p)
        fun_H_z!(ff.H_z, z, ff.H_z_temp, p)

        fun_dv!(dv_p, v_p, p.r, ff.x_lgn, ff.dv_temp, p)
        fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, ff.dv_temp, p)
        fun_dx_v1!(dx, ff.x, ff.C, z, p.x_V2, p)
        fun_dy!(dy, y, ff.C, ff.x, ff.m, ff.dy_temp, p)
        fun_dm!(dm, ff.m, ff.x, ff.dm_temp, p)
        fun_dz!(dz, z, y, ff.H_z, ff.s, ff.dz_temp, p)
        fun_ds!(ds, ff.s, ff.H_z, ff.ds_temp, p)

    end
    return nothing

end


mutable struct LamFunction_allStruct <: Function
	x::AbstractArray
	y::AbstractArray
	m::AbstractArray
	z::AbstractArray
	s::AbstractArray
	v_p::AbstractArray
	v_m::AbstractArray

	dx::AbstractArray
	dy::AbstractArray
	dm::AbstractArray
	dz::AbstractArray
	ds::AbstractArray
	dv_p::AbstractArray
	dv_m::AbstractArray

	x_lgn::AbstractArray
	C::AbstractArray
	H_z::AbstractArray

	dy_temp::AbstractArray
	dm_temp::AbstractArray
	dz_temp::AbstractArray
	ds_temp::AbstractArray
	dv_temp::AbstractArray
	H_z_temp::AbstractArray
	V_temp_1::AbstractArray
	V_temp_2::AbstractArray
	Q_temp::AbstractArray
	P_temp::AbstractArray
end


function (ff::LamFunction_allStruct)(du, u, p, t)

       @inbounds begin

		@. ff.x = @view u[:, :, 1:p.K,:]
        ff.y = @view u[:, :, p.K+1:2*p.K,:]
        @. ff.m = @view u[:, :, 2*p.K+1:3*p.K,:]
        ff.z = @view u[:, :, 3*p.K+1:4*p.K,:]
		@. ff.s = @view u[:, :, 4*p.K+1:5*p.K,:]

		ff.v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
		ff.v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

		ff.dx = @view du[:, :, 1:p.K,:]
        ff.dy = @view du[:, :, p.K+1:2*p.K,:]
        ff.dm = @view du[:, :, 2*p.K+1:3*p.K,:]
        ff.dz = @view du[:, :, 3*p.K+1:4*p.K,:]
        ff.ds = @view du[:, :, 4*p.K+1:5*p.K,:]

		ff.dv_p = @view du[:, :, 5*p.K+1:5*p.K+1,:]
		ff.dv_m = @view du[:, :, 5*p.K+2:5*p.K+2,:]

		fun_x_lgn!(ff.x_lgn, ff.x, p)
        fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.V_temp_1, ff.V_temp_2, ff.Q_temp, ff.P_temp, p)
        fun_H_z!(ff.H_z, ff.z, ff.H_z_temp, p)

        fun_dv!(ff.dv_p, ff.v_p, p.r, ff.x_lgn, ff.dv_temp, p)
        fun_dv!(ff.dv_m, ff.v_m, .-p.r, ff.x_lgn, ff.dv_temp, p)
        fun_dx_v1!(ff.dx, ff.x, ff.C, ff.z, p.x_V2, p)
        fun_dy!(ff.dy, ff.y, ff.C, ff.x, ff.m, ff.dy_temp, p)
        fun_dm!(ff.dm, ff.m, ff.x, ff.dm_temp, p)
        fun_dz!(ff.dz, ff.z, ff.y, ff.H_z, ff.s, ff.dz_temp, p)
        fun_ds!(ff.ds, ff.s, ff.H_z, ff.ds_temp, p)


    end
    return nothing

end



mutable struct LamFunction_gpu_reuse{T <: AbstractArray} <: Function

	x_lgn::T
	C::T
	H_z::T

	tmp_a::T
	tmp_b::T

	tmp_A::T
	tmp_B::T
	tmp_C::T
end


function (ff::LamFunction_gpu_reuse)(du, u, p, t)

       @inbounds begin


        y = @view u[:, :, p.K+1:2*p.K,:]
		z = @view u[:, :, 3*p.K+1:4*p.K,:]

		v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
		v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

		dx = @view du[:, :, 1:p.K,:]
        dy = @view du[:, :, p.K+1:2*p.K,:]
        dm = @view du[:, :, 2*p.K+1:3*p.K,:]
		dz = @view du[:, :, 3*p.K+1:4*p.K,:]
        ds = @view du[:, :, 4*p.K+1:5*p.K,:]

		dv_p = @view du[:, :, 5*p.K+1:5*p.K+1,:]
		dv_m = @view du[:, :, 5*p.K+2:5*p.K+2,:]


        fun_v_C!(ff.C, v_p, v_m, ff.tmp_a, ff.tmp_b, ff.tmp_A, ff.tmp_B, p) #a,b,C,D: buffers
        fun_H_z!(ff.H_z, z, ff.tmp_A, p) #A=buffer

		@. ff.tmp_A = @view u[:, :, 1:p.K,:] #x
        @. ff.tmp_B = @view u[:, :, 2*p.K+1:3*p.K,:] #m
		fun_x_lgn!(ff.x_lgn, ff.tmp_A, p)

        fun_dv!(dv_p, v_p, p.r, ff.x_lgn, ff.tmp_a, p) #a=buffer
        fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, ff.tmp_a, p) #a=buff


		fun_dx_v1!(dx, ff.tmp_A, ff.C, z, p.x_V2, p) #A:x
        fun_dy!(dy, y, ff.C, ff.tmp_A, ff.tmp_B, ff.tmp_C, p) # A:x, B:m, C:buffer
        fun_dm!(dm, ff.tmp_B, ff.tmp_A, ff.tmp_C, p) # A:x, B:m, C:buffer

		@. ff.tmp_A = @view u[:, :, 4*p.K+1:5*p.K,:] #s
		fun_dz!(dz, z, y, ff.H_z, ff.tmp_A, ff.tmp_B, p) #A=s, B=buffer
        fun_ds!(ds, ff.tmp_A, ff.H_z, ff.tmp_B, p) #A=s, B=buffer


    end
    return nothing

end

mutable struct LamFunction_equ{T <: AbstractArray} <: Function
	x::T
	m::T
	s::T

	x_lgn::T
	C::T
	H_z::T

	dy_temp::T
	dm_temp::T
	dz_temp::T
	ds_temp::T
	dv_temp::T
	H_z_temp::T
	V_temp_1::T
	V_temp_2::T
	Q_temp::T
	P_temp::T
end


function (ff::LamFunction_equ)(du, u, p, t)

       @inbounds begin

		@. ff.x = @view u[:, :, 1:p.K,:]
        y = @view u[:, :, p.K+1:2*p.K,:]
        @. ff.m = @view u[:, :, 2*p.K+1:3*p.K,:]
        z = @view u[:, :, 3*p.K+1:4*p.K,:]
		@. ff.s = @view u[:, :, 4*p.K+1:5*p.K,:]

		v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
		v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

		dx = @view du[:, :, 1:p.K,:]
        dy = @view du[:, :, p.K+1:2*p.K,:]
        dm = @view du[:, :, 2*p.K+1:3*p.K,:]
        dz = @view du[:, :, 3*p.K+1:4*p.K,:]
        ds = @view du[:, :, 4*p.K+1:5*p.K,:]

		dv_p = @view du[:, :, 5*p.K+1:5*p.K+1,:]
		dv_m = @view du[:, :, 5*p.K+2:5*p.K+2,:]

		fun_x_lgn!(ff.x_lgn, ff.x, p)
        fun_v_C!(ff.C, v_p, v_m, ff.V_temp_1, ff.V_temp_2, ff.Q_temp, ff.P_temp, p)
        fun_H_z!(ff.H_z, z, ff.H_z_temp, p)

        fun_dv!(dv_p, v_p, p.r, ff.x_lgn, ff.dv_temp, p)
        fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, ff.dv_temp, p)
#         fun_dx_v1!(dx, ff.x, ff.C, z, p.x_V2, p)
#         fun_dy!(dy, y, ff.C, ff.x, ff.m, ff.dy_temp, p)
		fun_x_equ!(ff.x, ff.C, z, ff.dy_temp, p.x_V2, p)
		fun_y_equ!(y, ff.C, ff.x, ff.m, ff.dy_temp, p)
        fun_dm!(dm, ff.m, ff.x, ff.dm_temp, p)
        fun_dz!(dz, z, y, ff.H_z, ff.s, ff.dz_temp, p)
        fun_ds!(ds, ff.s, ff.H_z, ff.ds_temp, p)

    end
    return nothing

end

mutable struct LamFunction_all_struct_reuse_1{T <: AbstractArray} <: Function

	x_lgn::T
	C::T
	H_z::T

	y
	z
	v_p
	v_m
	dx
	dy
	dm
	dz
	ds
	dv_p
	dv_m

	tmp_a::T
	tmp_b::T

	tmp_A::T
	tmp_B::T
	tmp_C::T
end


function (ff::LamFunction_all_struct_reuse_1)(du, u, p, t)

       @inbounds begin


        ff.y = @view u[:, :, p.K+1:2*p.K,:]
		ff.z = @view u[:, :, 3*p.K+1:4*p.K,:]

		ff.v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
		ff.v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

		ff.dx = @view du[:, :, 1:p.K,:]
        ff.dy = @view du[:, :, p.K+1:2*p.K,:]
        ff.dm = @view du[:, :, 2*p.K+1:3*p.K,:]
		ff.dz = @view du[:, :, 3*p.K+1:4*p.K,:]
        ff.ds = @view du[:, :, 4*p.K+1:5*p.K,:]

		ff.dv_p = @view du[:, :, 5*p.K+1:5*p.K+1,:]
		ff.dv_m = @view du[:, :, 5*p.K+2:5*p.K+2,:]


        fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.tmp_a, ff.tmp_b, ff.tmp_A, ff.tmp_B, p) #a,b,C,D: buffers
        fun_H_z!(ff.H_z, ff.z, ff.tmp_A, p) #A=buffer

		@. ff.tmp_A = @view u[:, :, 1:p.K,:] #x
        @. ff.tmp_B = @view u[:, :, 2*p.K+1:3*p.K,:] #m
		fun_x_lgn!(ff.x_lgn, ff.tmp_A, p)

        fun_dv!(ff.dv_p, ff.v_p, p.r, ff.x_lgn, ff.tmp_a, p) #a=buffer
        fun_dv!(ff.dv_m, ff.v_m, .-p.r, ff.x_lgn, ff.tmp_a, p) #a=buff


		fun_dx_v1!(ff.dx, ff.tmp_A, ff.C, ff.z, p.x_V2, p) #A:x
        fun_dy!(ff.dy, ff.y, ff.C, ff.tmp_A, ff.tmp_B, ff.tmp_C, p) # A:x, B:m, C:buffer
        fun_dm!(ff.dm, ff.tmp_B, ff.tmp_A, ff.tmp_C, p) # A:x, B:m, C:buffer

		@. ff.tmp_A = @view u[:, :, 4*p.K+1:5*p.K,:] #s
		fun_dz!(ff.dz, ff.z, ff.y, ff.H_z, ff.tmp_A, ff.tmp_B, p) #A=s, B=buffer
        fun_ds!(ff.ds, ff.tmp_A, ff.H_z, ff.tmp_B, p) #A=s, B=buffer


    end
    return nothing

end

# mutable struct LamFunction_gpu30 <: Function
# 	x_lgn::AbstractArray
# 	C::AbstractArray
# 	H_z::AbstractArray

# 	dy_temp::AbstractArray
# 	dm_temp::AbstractArray
# 	dz_temp::AbstractArray
# 	ds_temp::AbstractArray
# 	dv_temp::AbstractArray
# 	H_z_temp::AbstractArray
# 	V_temp_1::AbstractArray
# 	V_temp_2::AbstractArray
# 	Q_temp::AbstractArray
# 	P_temp::AbstractArray
# end


# function (ff::LamFunction_gpu30)(du, u, p, t)

#        @inbounds begin

# 		ff.x = @view u[:, :, 1:p.K,:]
#         ff.y = @view u[:, :, p.K+1:2*p.K,:]
#         @. ff.m = @view u[:, :, 2*p.K+1:3*p.K,:]
#         ff.z = @view u[:, :, 3*p.K+1:4*p.K,:]
# 		@. ff.s = @view u[:, :, 4*p.K+1:5*p.K,:]

# 		ff.v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
# 		ff.v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

# 		ff.dx = @view du[:, :, 1:p.K,:]
#         ff.dy = @view du[:, :, p.K+1:2*p.K,:]
#         ff.dm = @view du[:, :, 2*p.K+1:3*p.K,:]
#         ff.dz = @view du[:, :, 3*p.K+1:4*p.K,:]
#         ff.ds = @view du[:, :, 4*p.K+1:5*p.K,:]

# 		ff.dv_p = @view du[:, :, 5*p.K+1:5*p.K+1,:]
# 		ff.dv_m = @view du[:, :, 5*p.K+2:5*p.K+2,:]

# 		fun_x_lgn!(ff.x_lgn, ff.x, p)
#         fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.V_temp_1, ff.V_temp_2, ff.Q_temp, ff.P_temp, p)
#         fun_H_z!(ff.H_z, ff.z, ff.H_z_temp, p)

#         fun_dv!(ff.dv_p, ff.v_p, p.r, ff.x_lgn, ff.dv_temp, p)
#         fun_dv!(ff.dv_m, ff.v_m, .-p.r, ff.x_lgn, ff.dv_temp, p)
#         fun_dx_v1!(ff.dx, ff.x, ff.C, ff.z, p.x_V2, p)
#         fun_dy!(ff.dy, ff.y, ff.C, ff.x, ff.m, ff.dy_temp, p)
#         fun_dm!(ff.dm, ff.m, ff.x, ff.dm_temp, p)
#         fun_dz!(ff.dz, ff.z, ff.y, ff.H_z, ff.s, ff.dz_temp, p)
#         fun_ds!(ff.ds, ff.s, ff.H_z, ff.ds_temp, p)


#     end
#     return nothing

# end


# mutable struct LamFunction_gpu14{T <: CuArray{Float32,4,Nothing}} <: Function
# 	x::T
# 	y::T
# 	m::T
# 	z::T
# 	s::T
# 	v_p::T
# 	v_m::T

# # 	dx::T
# # 	dy::T
# # 	dm::T
# # 	dz::T
# # 	ds::T
# # 	dv_p::T
# # 	dv_m::T

# 	x_lgn::T
# 	C::T
# 	H_z::T

# 	dy_temp::T
# 	dm_temp::T
# 	dz_temp::T
# 	ds_temp::T
# 	dv_temp::T
# 	H_z_temp::T
# 	V_temp_1::T
# 	V_temp_2::T
# 	Q_temp::T
# 	P_temp::T
# end


# function (ff::LamFunction_gpu14)(du, u, p, t)

#        @inbounds begin
# #         @. ff.x = @view u[:, :, 1:p.K,:]
# #         @. ff.y = @view u[:, :, p.K+1:2*p.K,:]
#         @. ff.m = @view u[:, :, 2*p.K+1:3*p.K,:]
# #         @. ff.z = @view u[:, :, 3*p.K+1:4*p.K,:]
#         @. ff.s = @view u[:, :, 4*p.K+1:5*p.K,:]

# #        @. ff.v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
# # 		@. ff.v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

# 		x = @view u[:, :, 1:p.K,:]
#         y = @view u[:, :, p.K+1:2*p.K,:]
# #         m = @view u[:, :, 2*p.K+1:3*p.K,:]
#         z = @view u[:, :, 3*p.K+1:4*p.K,:]
# #         s = @view u[:, :, 4*p.K+1:5*p.K,:]

#        v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
# 		v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

# 		 dx = @view du[:, :, 1:p.K,:]
#         dy = @view du[:, :, p.K+1:2*p.K,:]
#         dm = @view du[:, :, 2*p.K+1:3*p.K,:]
#         dz = @view du[:, :, 3*p.K+1:4*p.K,:]
#         ds = @view du[:, :, 4*p.K+1:5*p.K,:]

#        dv_p = @view du[:, :, 5*p.K+1:5*p.K+1,:]
# 		dv_m = @view du[:, :, 5*p.K+2:5*p.K+2,:]

# #         fun_x_lgn!(ff.x_lgn, ff.x, p)
# #         fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.V_temp_1, ff.V_temp_2, ff.Q_temp, ff.P_temp, p)
# #         fun_H_z!(ff.H_z, ff.z, ff.H_z_temp, p)

# #         fun_dv!(dv_p, ff.v_p, p.r, ff.x_lgn, ff.dv_temp, p)
# #         fun_dv!(dv_m, ff.v_m, .-p.r, ff.x_lgn, ff.dv_temp, p)
# #         fun_dx_v1!(dx, ff.x, ff.C, ff.z, p.x_V2, p)
# #         fun_dy!(dy, ff.y, ff.C, ff.x, ff.m, ff.dy_temp, p)
# #         fun_dm!(dm, ff.m, ff.x, ff.dm_temp, p)
# #         fun_dz!(dz, ff.z, ff.y, ff.H_z, ff.s, ff.dz_temp, p)
# #         fun_ds!(ds, ff.s, ff.H_z, ff.ds_temp, p)

# # 		fun_x_lgn!(ff.x_lgn, x, p)
#         fun_v_C!(ff.C, v_p, v_m, ff.V_temp_1, ff.V_temp_2, ff.Q_temp, ff.P_temp, p)
#         fun_H_z!(ff.H_z, z, ff.H_z_temp, p)

#         fun_dv!(dv_p, v_p, p.r, ff.x_lgn, ff.dv_temp, p)
#         fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, ff.dv_temp, p)
#         fun_dx_v1!(dx, x, ff.C, z, p.x_V2, p)
#         fun_dy!(dy, y, ff.C, x, ff.m, ff.dy_temp, p)
#         fun_dm!(dm, ff.m, x, ff.dm_temp, p)
#         fun_dz!(dz, z, y, ff.H_z, ff.s, ff.dz_temp, p)
#         fun_ds!(ds, ff.s, ff.H_z, ff.ds_temp, p)


# # 		@. du[:, :, 1:p.K,:] = ff.dx
# # 		@. du[:, :, p.K+1:2*p.K,:] = ff.dy
# # 		@. du[:, :, 2*p.K+1:3*p.K,:] = ff.dm
# # 		@. du[:, :, 3*p.K+1:4*p.K,:] = ff.dz
# # 		@. du[:, :, 4*p.K+1:5*p.K,:] = ff.ds

# #         @. du[:, :, 5*p.K+1:5*p.K+1,:] = ff.dv_p
# # 		@. du[:, :, 5*p.K+2:5*p.K+2,:] = ff.dv_m

#     end
#     return nothing

# end



# mutable struct LamFunction_gpu14{T <: CuArray{Float32,4,Nothing}} <: Function
# 	x::T
# 	y::T
# 	m::T
# 	z::T
# 	s::T
# 	v_p::T
# 	v_m::T

# # 	dx::T
# # 	dy::T
# # 	dm::T
# # 	dz::T
# # 	ds::T
# # 	dv_p::T
# # 	dv_m::T

# 	x_lgn::T
# 	C::T
# 	H_z::T

# 	dy_temp::T
# 	dm_temp::T
# 	dz_temp::T
# 	ds_temp::T
# 	dv_temp::T
# 	H_z_temp::T
# 	V_temp_1::T
# 	V_temp_2::T
# 	Q_temp::T
# 	P_temp::T
# end


# function (ff::LamFunction_gpu14)(du, u, p, t)

#        @inbounds begin
# #         @. ff.x = @view u[:, :, 1:p.K,:]
# #         @. ff.y = @view u[:, :, p.K+1:2*p.K,:]
#         @. ff.m = @view u[:, :, 2*p.K+1:3*p.K,:]
# #         @. ff.z = @view u[:, :, 3*p.K+1:4*p.K,:]
#         @. ff.s = @view u[:, :, 4*p.K+1:5*p.K,:]

# #        @. ff.v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
# # 		@. ff.v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

# 		x = @view u[:, :, 1:p.K,:]
#         y = @view u[:, :, p.K+1:2*p.K,:]
# #         m = @view u[:, :, 2*p.K+1:3*p.K,:]
#         z = @view u[:, :, 3*p.K+1:4*p.K,:]
# #         s = @view u[:, :, 4*p.K+1:5*p.K,:]

#        v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
# 		v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

# 		 dx = @view du[:, :, 1:p.K,:]
#         dy = @view du[:, :, p.K+1:2*p.K,:]
#         dm = @view du[:, :, 2*p.K+1:3*p.K,:]
#         dz = @view du[:, :, 3*p.K+1:4*p.K,:]
#         ds = @view du[:, :, 4*p.K+1:5*p.K,:]

#        dv_p = @view du[:, :, 5*p.K+1:5*p.K+1,:]
# 		dv_m = @view du[:, :, 5*p.K+2:5*p.K+2,:]

# #         fun_x_lgn!(ff.x_lgn, ff.x, p)
# #         fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.V_temp_1, ff.V_temp_2, ff.Q_temp, ff.P_temp, p)
# #         fun_H_z!(ff.H_z, ff.z, ff.H_z_temp, p)

# #         fun_dv!(dv_p, ff.v_p, p.r, ff.x_lgn, ff.dv_temp, p)
# #         fun_dv!(dv_m, ff.v_m, .-p.r, ff.x_lgn, ff.dv_temp, p)
# #         fun_dx_v1!(dx, ff.x, ff.C, ff.z, p.x_V2, p)
# #         fun_dy!(dy, ff.y, ff.C, ff.x, ff.m, ff.dy_temp, p)
# #         fun_dm!(dm, ff.m, ff.x, ff.dm_temp, p)
# #         fun_dz!(dz, ff.z, ff.y, ff.H_z, ff.s, ff.dz_temp, p)
# #         fun_ds!(ds, ff.s, ff.H_z, ff.ds_temp, p)

# # 		fun_x_lgn!(ff.x_lgn, x, p)
#         fun_v_C!(ff.C, v_p, v_m, ff.V_temp_1, ff.V_temp_2, ff.Q_temp, ff.P_temp, p)
#         fun_H_z!(ff.H_z, z, ff.H_z_temp, p)

#         fun_dv!(dv_p, v_p, p.r, ff.x_lgn, ff.dv_temp, p)
#         fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, ff.dv_temp, p)
#         fun_dx_v1!(dx, x, ff.C, z, p.x_V2, p)
#         fun_dy!(dy, y, ff.C, x, ff.m, ff.dy_temp, p)
#         fun_dm!(dm, ff.m, x, ff.dm_temp, p)
#         fun_dz!(dz, z, y, ff.H_z, ff.s, ff.dz_temp, p)
#         fun_ds!(ds, ff.s, ff.H_z, ff.ds_temp, p)


# # 		@. du[:, :, 1:p.K,:] = ff.dx
# # 		@. du[:, :, p.K+1:2*p.K,:] = ff.dy
# # 		@. du[:, :, 2*p.K+1:3*p.K,:] = ff.dm
# # 		@. du[:, :, 3*p.K+1:4*p.K,:] = ff.dz
# # 		@. du[:, :, 4*p.K+1:5*p.K,:] = ff.ds

# #         @. du[:, :, 5*p.K+1:5*p.K+1,:] = ff.dv_p
# # 		@. du[:, :, 5*p.K+2:5*p.K+2,:] = ff.dv_m

#     end
#     return nothing

# end





# mutable struct LamFunction_gpu2{T <: CuArray{Float32,4,Nothing}} <: Function
# 	x::T
# 	y::T
# 	m::T
# 	z::T
# 	s::T
# 	v_p::T
# 	v_m::T

# 	dx::T
# 	dy::T
# 	dm::T
# 	dz::T
# 	ds::T
# 	dv_p::T
# 	dv_m::T

# 	x_lgn::T
# 	C::T
# 	H_z::T
# 	dy_temp::T
# 	ds_temp::T
# 	H_z_temp::T
# 	V_temp_1::T
# 	V_temp_2::T
# 	Q_temp::T
# 	P_temp::T
# end

# mutable struct LamFunction_cpu2{T <: AbstractArray} <: Function
# 	x::T
# 	y::T
# 	m::T
# 	z::T
# 	s::T
# 	v_p::T
# 	v_m::T

# 	dx::T
# 	dy::T
# 	dm::T
# 	dz::T
# 	ds::T
# 	dv_p::T
# 	dv_m::T

# 	x_lgn::T
# 	C::T
# 	H_z::T
# 	ds_temp::T
# 		H_z_temp::T
# 	V_temp_1::T
# 	V_temp_2::T
# 	Q_temp::T
# 	P_temp::T
# end

# mutable struct LamFunction{T} <: Function
# 	x::AbstractArray
# 	y::AbstractArray
# 	m::AbstractArray
# 	z::AbstractArray
# 	s::AbstractArray
# 	v_p::AbstractArray
# 	v_m::AbstractArray

# 	dx::AbstractArray
# 	dy::AbstractArray
# 	dm::AbstractArray
# 	dz::AbstractArray
# 	ds::AbstractArray
# 	dv_p::AbstractArray
# 	dv_m::AbstractArray

# 	x_lgn::AbstractArray
# 	C::AbstractArray
# 	H_z::AbstractArray
# 	V_temp_1::AbstractArray
# 	V_temp_2::AbstractArray
# 	Q_temp::AbstractArray
# 	P_temp::AbstractArray
# end

# function (ff::LamFunction_cpu2)(du, u, p, t)

#     @inbounds begin
#         @. ff.x = @view u[:, :, 1:p.K,:]
#         @. ff.y = @view u[:, :, p.K+1:2*p.K,:]
#         @. ff.m = @view u[:, :, 2*p.K+1:3*p.K,:]
#         @. ff.z = @view u[:, :, 3*p.K+1:4*p.K,:]
#         @. ff.s = @view u[:, :, 4*p.K+1:5*p.K,:]

#         @. ff.v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
# 		@. ff.v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

#         fun_x_lgn!(ff.x_lgn, ff.x, p)
#         fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.V_temp_1, ff.V_temp_2, ff.Q_temp, ff.P_temp, p)
#         fun_H_z!(ff.H_z, ff.z, ff.H_z_temp, p)

#         fun_dv!(ff.dv_p, ff.v_p, p.r, ff.x_lgn, p)
#         fun_dv!(ff.dv_m, ff.v_m, .-p.r, ff.x_lgn, p)
#         fun_dx_v1!(ff.dx, ff.x, ff.C, ff.z, p.x_V2, p)
#         fun_dy!(ff.dy, ff.y, ff.C, ff.x, ff.m, p)
#         fun_dm!(ff.dm, ff.m, ff.x, p)
#         fun_dz!(ff.dz, ff.z, ff.y, ff.H_z, ff.s, p)
#         fun_ds!(ff.ds, ff.s, ff.H_z, ff.ds_temp, p)

# 		@. du[:, :, 1:p.K,:] = ff.dx
# 		@. du[:, :, p.K+1:2*p.K,:] = ff.dy
# 		@. du[:, :, 2*p.K+1:3*p.K,:] = ff.dm
# 		@. du[:, :, 3*p.K+1:4*p.K,:] = ff.dz
# 		@. du[:, :, 4*p.K+1:5*p.K,:] = ff.ds

#         @. du[:, :, 5*p.K+1:5*p.K+1,:] = ff.dv_p
# 		@. du[:, :, 5*p.K+2:5*p.K+2,:] = ff.dv_m

#     end
#     return nothing
# end


# function (ff::LamFunction_gpu2)(du, u, p, t)

#        @inbounds begin
#         @. ff.x = @view u[:, :, 1:p.K,:]
#         @. ff.y = @view u[:, :, p.K+1:2*p.K,:]
#         @. ff.m = @view u[:, :, 2*p.K+1:3*p.K,:]
#         @. ff.z = @view u[:, :, 3*p.K+1:4*p.K,:]
#         @. ff.s = @view u[:, :, 4*p.K+1:5*p.K,:]

#         @. ff.v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
# 		@. ff.v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

#         fun_x_lgn!(ff.x_lgn, ff.x, p)
#         fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.V_temp_1, ff.V_temp_2, ff.Q_temp, ff.P_temp, p)
#         fun_H_z!(ff.H_z, ff.z, ff.H_z_temp, p)

#         fun_dv!(ff.dv_p, ff.v_p, p.r, ff.x_lgn, p)
#         fun_dv!(ff.dv_m, ff.v_m, .-p.r, ff.x_lgn, p)
#         fun_dx_v1!(ff.dx, ff.x, ff.C, ff.z, p.x_V2, p)
#         fun_dy!(ff.dy, ff.y, ff.C, ff.x, ff.m, ff.dy_temp, p)
#         fun_dm!(ff.dm, ff.m, ff.x, p)
#         fun_dz!(ff.dz, ff.z, ff.y, ff.H_z, ff.s, p)
#         fun_ds!(ff.ds, ff.s, ff.H_z, ff.ds_temp, p)

# 		@. du[:, :, 1:p.K,:] = ff.dx
# 		@. du[:, :, p.K+1:2*p.K,:] = ff.dy
# 		@. du[:, :, 2*p.K+1:3*p.K,:] = ff.dm
# 		@. du[:, :, 3*p.K+1:4*p.K,:] = ff.dz
# 		@. du[:, :, 4*p.K+1:5*p.K,:] = ff.ds

#         @. du[:, :, 5*p.K+1:5*p.K+1,:] = ff.dv_p
# 		@. du[:, :, 5*p.K+2:5*p.K+2,:] = ff.dv_m

#     end
#     return nothing

# end





# mutable struct LamFunction_04{T} <: Function
# x::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# y::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# m::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# z::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# s::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# v_p::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# v_m::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# dx::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# dy::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# dm::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# dz::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# ds::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# dv_p::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# dv_m::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# 	x_lgn::CuArray{Float32,4,Nothing}
# 	C::CuArray{Float32,4,Nothing}
# 	H_z::CuArray{Float32,4,Nothing}
# 	V_temp_1::CuArray{Float32,4,Nothing}
# 	V_temp_2::CuArray{Float32,4,Nothing}
# 	Q_temp::CuArray{Float32,4,Nothing}
# 	P_temp::CuArray{Float32,4,Nothing}
# end

# mutable struct LamFunction_10 <: Function
# 	x
# 	y
# 	m
# 	z
# 	s
# 	v_p
# 	v_m
# 	dx
# 	dy
# 	dm
# 	dz
# 	ds
# 	dv_p
# 	dv_m
# 	x_lgn
# 	C
# 	H_z
# 	V_temp_1
# 	V_temp_2
# 	Q_temp
# 	P_temp
# end

# mutable struct LamFunction_13<: Function
# 	x::CuArray{Float32,4,Nothing}
# 	y::CuArray{Float32,4,Nothing}
# 	m::CuArray{Float32,4,Nothing}
# 	z::CuArray{Float32,4,Nothing}
# 	s::CuArray{Float32,4,Nothing}
# 	v_p::CuArray{Float32,4,Nothing}
# 	v_m::CuArray{Float32,4,Nothing}
# dx::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# 	dy::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# 	dm::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# 	dz::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# 	ds::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# 	dv_p::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# 	dv_m::SubArray{Float32,4,CuArray{Float32,4,Nothing},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}
# 	x_lgn::CuArray{Float32,4,Nothing}
# 	C::CuArray{Float32,4,Nothing}
# 	H_z::CuArray{Float32,4,Nothing}
# 	V_temp_1::CuArray{Float32,4,Nothing}
# 	V_temp_2::CuArray{Float32,4,Nothing}
# 	Q_temp::CuArray{Float32,4,Nothing}
# 	P_temp::CuArray{Float32,4,Nothing}
# end

# function (ff::LamFunction_13)(du, u, p, t)

#     @inbounds begin
# #         ff.x = @view u[:, :, 1:p.K,:]
# #         ff.y = @view u[:, :, p.K+1:2*p.K,:]
# #         ff.m = @view u[:, :, 2*p.K+1:3*p.K,:]
# #         ff.z = @view u[:, :, 3*p.K+1:4*p.K,:]
# #         ff.s = @view u[:, :, 4*p.K+1:5*p.K,:]

# #         ff.v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
# # 		ff.v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

# 		@. ff.x = @view u[:, :, 1:p.K,:]
#         @. ff.y = @view u[:, :, p.K+1:2*p.K,:]
#         @. ff.m = @view u[:, :, 2*p.K+1:3*p.K,:]
#         @. ff.z = @view u[:, :, 3*p.K+1:4*p.K,:]
#         @. ff.s = @view u[:, :, 4*p.K+1:5*p.K,:]

#         @. ff.v_p = @view u[:, :, 5*p.K+1:5*p.K+1,:]
# 		@. ff.v_m = @view u[:, :, 5*p.K+2:5*p.K+2,:]

# # 		ff.dx = @view du[:, :, 1:p.K,:]
# # 		ff.dy = @view du[:, :, p.K+1:2*p.K,:]
# # 		ff.dm = @view du[:, :, 2*p.K+1:3*p.K,:]
# # 		ff.dz = @view du[:, :, 3*p.K+1:4*p.K,:]
# # 		ff.ds = @view du[:, :, 4*p.K+1:5*p.K,:]

# # 		ff.dv_p = @view du[:, :, 5*p.K+1:5*p.K+1,:]
# # 		ff.dv_m = @view du[:, :, 5*p.K+2:5*p.K+2,:]

# # 		ff.x_lgn = @view ff.x_lgn[:,:,:,:]
# # 		ff.C_ = @view ff.C[:,:,:,:]
# # 		ff.H_z_ = @view ff.H_z[:,:,:,:]
# # 		ff.V_temp_1_ = @view ff.V_temp_1[:,:,:,:]
# # 		ff.V_temp_2_ = @view ff.V_temp_2[:,:,:,:]
# # 		ff.Q_temp_ = @view ff.Q_temp[:,:,:,:]
# # 		ff.P_temp_ = @view ff.P_temp[:,:,:,:]

#         fun_x_lgn!(ff.x_lgn, ff.x, p)
#         fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.V_temp_1, ff.V_temp_2, ff.Q_temp, ff.P_temp, p)
#         fun_H_z!(ff.H_z, ff.z, p)

#         fun_dv!(ff.dv_p, ff.v_p, p.r, ff.x_lgn, p)
#         fun_dv!(ff.dv_m, ff.v_m, .-p.r, ff.x_lgn, p)
#         fun_dx_v1!(ff.dx, ff.x, ff.C, ff.z, p.x_V2, p)
#         fun_dy!(ff.dy, ff.y, ff.C, ff.x, ff.m, ff.dy_temp, p)
#         fun_dm!(ff.dm, ff.m, ff.x, p)
#         fun_dz!(ff.dz, ff.z, ff.y, ff.H_z, ff.s, p)
#         fun_ds!(ff.ds, ff.s, ff.H_z, p)

# # 		@. du[:, :, 1:p.K,:] = ff.dx
# # 		@. du[:, :, p.K+1:2*p.K,:] = ff.dy
# # 		@. du[:, :, 2*p.K+1:3*p.K,:] = ff.dm
# # 		@. du[:, :, 3*p.K+1:4*p.K,:] = ff.dz
# # 		@. du[:, :, 4*p.K+1:5*p.K,:] = ff.ds

# #         @. du[:, :, 5*p.K+1:5*p.K+1,:] = ff.dv_p
# # 		@. du[:, :, 5*p.K+2:5*p.K+2,:] = ff.dv_m

#     end
#     return nothing
# end




function kernels_gpu(img::AbstractArray, p::NamedTuple)
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

temp_out = (
        k_gauss_1 = cu(reshape2d_4d(Kernel.gaussian(p.σ_1))),
        k_gauss_2 = cu(reshape2d_4d(Kernel.gaussian(p.σ_2))),
        k_C_d = cu(C_Q_temp),
        k_C_b = cu(C_P_temp),

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

function kernels_cpu(img::AbstractArray, p::NamedTuple)
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


	temp_out = (
        k_gauss_1 = reshape2d_4d(Kernel.gaussian(p.σ_1)),
        k_gauss_2 = reshape2d_4d(Kernel.gaussian(p.σ_2)),
        k_C_d = C_Q_temp,
        k_C_b = C_P_temp,

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




function reshape2d_4d(img::AbstractArray)
    reshape(img, size(img)[1], size(img)[2], 1, 1)
end

# function reshape_ijk_ij1k(img::AbstractArray, p::NamedTuple)
#     reshape(img, size(img)[1], size(img)[2], 1, p.K)
# end
	
	
function conv!(out::AbstractArray, img::AbstractArray, kern::AbstractArray, p::NamedTuple)
# 	out_ = @view out[:,:,:,:]
# 	img_ = @view img[:,:,:,:]
# 	kern_ = @view kern[:,:,:,:]
# 	    @inbounds out .= NNlib.conv(img[:,:,:,:], kern, pad=(size(kern)[1]>>1, size(kern)[1]>>1, size(kern)[2]>>1, size(kern)[2]>>1), flipped=true)

#     @inbounds out .= NNlib.conv(img, kern, pad=(size(kern)[1]>>1, size(kern)[1]>>1, size(kern)[2]>>1, size(kern)[2]>>1), flipped=true)
# 	    @inbounds out .= NNlib.conv(img_, kern_, pad=(size(kern_)[1]>>1, size(kern_)[1]>>1, size(kern_)[2]>>1, size(kern_)[2]>>1), flipped=true)
    @inbounds NNlib.conv!(out, img, kern, NNlib.DenseConvDims(img, kern, padding = ( size(kern)[1]>>1 , size(kern)[1]>>1 , size(kern)[2]>>1 , size(kern)[2]>>1 ), flipkernel = true))

# 	@. out_ = out
    return nothing
end

# function conv!(out::AbstractArray, img::AbstractArray, kern::AbstractArray, p::NamedTuple)
# # 	out_ = @view out[:,:,:,:]
#     @inbounds out = NNlib.conv(img, kern, pad=(size(kern)[1]>>1, size(kern)[1]>>1, size(kern)[2]>>1, size(kern)[2]>>1), flipped=true)
# # 	@. out_ = out
#     return nothing
# end

function add_I_u_p(I::AbstractArray, p::NamedTuple)
# 	todo fix
# 	I_4d = cu(reshape2d_4d(I))
	I_4d = reshape2d_4d(I)
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

# function fun_x_lgn!(x_lgn::AbstractArray, x::AbstractArray, p::NamedTuple)
# # 	out_ = @view x_lgn[:,:,:,:]
# 	    @inbounds x_lgn .= NNlib.conv(x[:,:,:,:], p.k_x_lgn, pad=0,flipped=true)
#     return nothing
# end


function fun_x_lgn!(x_lgn::AbstractArray, x::AbstractArray, p::NamedTuple)
# 	out_ = @view x_lgn[:,:,:,:]
# 	x_ = @view x[:,:,:,:]
# 	kern_ = @view p.k_x_lgn[:,:,:,:]
# 	    @inbounds x_lgn .= NNlib.conv(x_, p.k_x_lgn, pad=0,flipped=true)
	
		    @inbounds NNlib.conv!(x_lgn, x, p.k_x_lgn, NNlib.DenseConvDims(x, p.k_x_lgn, padding=0, flipkernel=true))
	
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




function fun_f!(arr::AbstractArray, p::NamedTuple)
   @inbounds @. arr = (p.μ * arr^p.n) / (p.ν_pw_n + arr^p.n)
    return nothing
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


# lgn to l6 and l4
# function fun_v_C!(
#     v_C::AbstractArray,
#     v_p::AbstractArray,
#     v_m::AbstractArray, 
# 	V_temp_1::AbstractArray,
# 	V_temp_2::AbstractArray,
# 	Q_temp::AbstractArray,
# 	P_temp::AbstractArray,
#     p::NamedTuple,
# )
# #     V = similar(v_p)
# #     temp = similar(v_p)

# # 	A = similar(v_C)
# # 	B = similar(v_C)
#     #     allocate B to v_C
	
# # 	mx_v_p = similar(v_p)
	
# # 	@. mx_v_p = max(v_p,0f0)
	
#     @inbounds @. V_temp_2 = exp(-1.0f0 / 8.0f0) * (max(v_p[:,:,:,:], 0f0) - max(v_m[:,:,:,:], 0f0))
#     conv!(V_temp_1, V_temp_2, p.k_gauss_2, p)

   
    
# 	conv!(Q_temp, V_temp_1, p.k_C_d, p)
# 	conv!(P_temp, V_temp_1, p.k_C_b, p)
# # 	@. P_temp = abs(v_C)
# 	@inbounds @. P_temp = abs(v_C[:,:,:,:])
	
	
#     @inbounds @. v_C = p.γ * (max(Q_temp[:,:,:,:] - P_temp[:,:,:,:], 0f0) + max(-Q_temp[:,:,:,:] - P_temp[:,:,:,:], 0f0))
	
	
#     return nothing
# end

function fun_v_C!(
    v_C::AbstractArray,
    v_p::AbstractArray,
    v_m::AbstractArray, 
	V_temp_1::AbstractArray,
	V_temp_2::AbstractArray,
	Q_temp::AbstractArray,
	P_temp::AbstractArray,
    p::NamedTuple,
)
#     V = similar(v_p)
#     temp = similar(v_p)

# 	A = similar(v_C)
# 	B = similar(v_C)
    #     allocate B to v_C
	
# 	mx_v_p = similar(v_p)
	
# 	@. mx_v_p = max(v_p,0f0)
	
    @inbounds begin
		@. V_temp_2 = exp(-1.0f0 / 8.0f0) * (max(v_p, 0f0) - max(v_m, 0f0))
    conv!(V_temp_1, V_temp_2, p.k_gauss_2, p)

   
    
	conv!(Q_temp, V_temp_1, p.k_C_d, p)
	conv!(P_temp, V_temp_1, p.k_C_b, p)
	@. P_temp = abs(P_temp)
	
	
    @. v_C = p.γ * (max(Q_temp - P_temp, 0f0) + max(-Q_temp - P_temp, 0f0))
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
function fun_y_equ!(y::AbstractArray, C::AbstractArray, x::AbstractArray,  m::AbstractArray, dy_temp::AbstractArray, p::NamedTuple)
	@inbounds begin
		conv!(dy_temp, m, p.k_W_p, p)
		@. dy_temp = m * dy_temp
		fun_f!(dy_temp, p)
		@. y = (C + (p.η_p * x) - dy_temp)/(1 + C + (p.η_p * x) + dy_temp)
	end
    return nothing
end



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