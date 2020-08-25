"""
# module LaminartFunc

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-08-20


Functions for DiffEq to call LAMINART equations and hold values.
# Examples

```jldoctest
julia>
```
"""
module LaminartFunc

include("./LaminartKernels.jl")
include("./LaminartEqImfilter.jl")
include("./LaminartEqImfilterGPU_FFT.jl")
include("./LaminartEqImfilterGPU_FIR.jl")
include("./LaminartEqImfilterGPU_IIR.jl")
include("./LaminartEqConv.jl")


using NNlib, ImageFiltering, Images, OffsetArrays, CUDA



mutable struct LamFunction{T<:AbstractArray} <: Function
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
    A_temp::T
    B_temp::T
end

function (ff::LamFunction)(du, u, p, t)

    @inbounds begin

        @. ff.x = @view u[:, :, 1:p.K, :]
        y = @view u[:, :, p.K+1:2*p.K, :]
        @. ff.m = @view u[:, :, 2*p.K+1:3*p.K, :]
        z = @view u[:, :, 3*p.K+1:4*p.K, :]
        @. ff.s = @view u[:, :, 4*p.K+1:5*p.K, :]

        v_p = @view u[:, :, 5*p.K+1:5*p.K+1, :]
        v_m = @view u[:, :, 5*p.K+2:5*p.K+2, :]

        dx = @view du[:, :, 1:p.K, :]
        dy = @view du[:, :, p.K+1:2*p.K, :]
        dm = @view du[:, :, 2*p.K+1:3*p.K, :]
        dz = @view du[:, :, 3*p.K+1:4*p.K, :]
        ds = @view du[:, :, 4*p.K+1:5*p.K, :]

        dv_p = @view du[:, :, 5*p.K+1:5*p.K+1, :]
        dv_m = @view du[:, :, 5*p.K+2:5*p.K+2, :]

        LaminartEqConv.fun_x_lgn!(ff.x_lgn, ff.x, p)
        LaminartEqConv.fun_v_C!(
            ff.C,
            v_p,
            v_m,
            ff.V_temp_1,
            ff.V_temp_2,
            ff.A_temp,
            ff.B_temp,
            p,
        )
        LaminartEqConv.fun_H_z!(ff.H_z, z, ff.H_z_temp, p)

        LaminartEqConv.fun_dv!(dv_p, v_p, p.r, ff.x_lgn, ff.dv_temp, p)
        LaminartEqConv.fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, ff.dv_temp, p)
        LaminartEqConv.fun_dx_v1!(dx, ff.x, ff.C, z, p.x_V2, p)
        LaminartEqConv.fun_dy!(dy, y, ff.C, ff.x, ff.m, ff.dy_temp, p)
        LaminartEqConv.fun_dm!(dm, ff.m, ff.x, ff.dm_temp, p)
        LaminartEqConv.fun_dz!(dz, z, y, ff.H_z, ff.s, ff.dz_temp, p)
        LaminartEqConv.fun_ds!(ds, ff.s, ff.H_z, ff.ds_temp, p)

    end
    return nothing

end

"""
All arrays held in struct.
"""
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
    A_temp::AbstractArray
    B_temp::AbstractArray
end


function (ff::LamFunction_allStruct)(du, u, p, t)

    @inbounds begin

        @. ff.x = @view u[:, :, 1:p.K, :]
        ff.y = @view u[:, :, p.K+1:2*p.K, :]
        @. ff.m = @view u[:, :, 2*p.K+1:3*p.K, :]
        ff.z = @view u[:, :, 3*p.K+1:4*p.K, :]
        @. ff.s = @view u[:, :, 4*p.K+1:5*p.K, :]

        ff.v_p = @view u[:, :, 5*p.K+1:5*p.K+1, :]
        ff.v_m = @view u[:, :, 5*p.K+2:5*p.K+2, :]

        ff.dx = @view du[:, :, 1:p.K, :]
        ff.dy = @view du[:, :, p.K+1:2*p.K, :]
        ff.dm = @view du[:, :, 2*p.K+1:3*p.K, :]
        ff.dz = @view du[:, :, 3*p.K+1:4*p.K, :]
        ff.ds = @view du[:, :, 4*p.K+1:5*p.K, :]

        ff.dv_p = @view du[:, :, 5*p.K+1:5*p.K+1, :]
        ff.dv_m = @view du[:, :, 5*p.K+2:5*p.K+2, :]

        LaminartEqConv.fun_x_lgn!(ff.x_lgn, ff.x, p)
        LaminartEqConv.fun_v_C!(
            ff.C,
            ff.v_p,
            ff.v_m,
            ff.V_temp_1,
            ff.V_temp_2,
            ff.A_temp,
            ff.B_temp,
            p,
        )
        LaminartEqConv.fun_H_z!(ff.H_z, ff.z, ff.H_z_temp, p)

        LaminartEqConv.fun_dv!(ff.dv_p, ff.v_p, p.r, ff.x_lgn, ff.dv_temp, p)
        LaminartEqConv.fun_dv!(ff.dv_m, ff.v_m, .-p.r, ff.x_lgn, ff.dv_temp, p)
        LaminartEqConv.fun_dx_v1!(ff.dx, ff.x, ff.C, ff.z, p.x_V2, p)
        LaminartEqConv.fun_dy!(ff.dy, ff.y, ff.C, ff.x, ff.m, ff.dy_temp, p)
        LaminartEqConv.fun_dm!(ff.dm, ff.m, ff.x, ff.dm_temp, p)
        LaminartEqConv.fun_dz!(ff.dz, ff.z, ff.y, ff.H_z, ff.s, ff.dz_temp, p)
        LaminartEqConv.fun_ds!(ff.ds, ff.s, ff.H_z, ff.ds_temp, p)
    end
    return nothing

end


"""
Temp arrays resused for differiant function.
"""
mutable struct LamFunction_gpu_reuse{T<:AbstractArray} <: Function

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


        y = @view u[:, :, p.K+1:2*p.K, :]
        z = @view u[:, :, 3*p.K+1:4*p.K, :]

        v_p = @view u[:, :, 5*p.K+1:5*p.K+1, :]
        v_m = @view u[:, :, 5*p.K+2:5*p.K+2, :]

        dx = @view du[:, :, 1:p.K, :]
        dy = @view du[:, :, p.K+1:2*p.K, :]
        dm = @view du[:, :, 2*p.K+1:3*p.K, :]
        dz = @view du[:, :, 3*p.K+1:4*p.K, :]
        ds = @view du[:, :, 4*p.K+1:5*p.K, :]

        dv_p = @view du[:, :, 5*p.K+1:5*p.K+1, :]
        dv_m = @view du[:, :, 5*p.K+2:5*p.K+2, :]


        LaminartEqConv.fun_v_C!(
            ff.C,
            v_p,
            v_m,
            ff.tmp_a,
            ff.tmp_b,
            ff.tmp_A,
            ff.tmp_B,
            p,
        ) #a,b,C,D: buffers
        LaminartEqConv.fun_H_z!(ff.H_z, z, ff.tmp_A, p) #A=buffer

        @. ff.tmp_A = @view u[:, :, 1:p.K, :] #x
        @. ff.tmp_B = @view u[:, :, 2*p.K+1:3*p.K, :] #m
        LaminartEqConv.fun_x_lgn!(ff.x_lgn, ff.tmp_A, p)

        LaminartEqConv.fun_dv!(dv_p, v_p, p.r, ff.x_lgn, ff.tmp_a, p) #a=buffer
        LaminartEqConv.fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, ff.tmp_a, p) #a=buff


        LaminartEqConv.fun_dx_v1!(dx, ff.tmp_A, ff.C, z, p.x_V2, p) #A:x
        LaminartEqConv.fun_dy!(dy, y, ff.C, ff.tmp_A, ff.tmp_B, ff.tmp_C, p) # A:x, B:m, C:buffer
        LaminartEqConv.fun_dm!(dm, ff.tmp_B, ff.tmp_A, ff.tmp_C, p) # A:x, B:m, C:buffer

        @. ff.tmp_A = @view u[:, :, 4*p.K+1:5*p.K, :] #s
        LaminartEqConv.fun_dz!(dz, z, y, ff.H_z, ff.tmp_A, ff.tmp_B, p) #A=s, B=buffer
        LaminartEqConv.fun_ds!(ds, ff.tmp_A, ff.H_z, ff.tmp_B, p) #A=s, B=buffer


    end
    return nothing

end


"""
Uses equations solved at equilibrum for L6 (x) and L4 (y).
"""
mutable struct LamFunction_equ{T<:AbstractArray} <: Function
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
    A_temp::T
    B_temp::T
end


function (ff::LamFunction_equ)(du, u, p, t)

    @inbounds begin

        @. ff.x = @view u[:, :, 1:p.K, :]
        y = @view u[:, :, p.K+1:2*p.K, :]
        @. ff.m = @view u[:, :, 2*p.K+1:3*p.K, :]
        z = @view u[:, :, 3*p.K+1:4*p.K, :]
        @. ff.s = @view u[:, :, 4*p.K+1:5*p.K, :]

        v_p = @view u[:, :, 5*p.K+1:5*p.K+1, :]
        v_m = @view u[:, :, 5*p.K+2:5*p.K+2, :]

        dx = @view du[:, :, 1:p.K, :]
        dy = @view du[:, :, p.K+1:2*p.K, :]
        dm = @view du[:, :, 2*p.K+1:3*p.K, :]
        dz = @view du[:, :, 3*p.K+1:4*p.K, :]
        ds = @view du[:, :, 4*p.K+1:5*p.K, :]

        dv_p = @view du[:, :, 5*p.K+1:5*p.K+1, :]
        dv_m = @view du[:, :, 5*p.K+2:5*p.K+2, :]

        LaminartEqConv.fun_x_lgn!(ff.x_lgn, ff.x, p)
        LaminartEqConv.fun_v_C!(
            ff.C,
            v_p,
            v_m,
            ff.V_temp_1,
            ff.V_temp_2,
            ff.A_temp,
            ff.B_temp,
            p,
        )
        LaminartEqConv.fun_H_z!(ff.H_z, z, ff.H_z_temp, p)

        LaminartEqConv.fun_dv!(dv_p, v_p, p.r, ff.x_lgn, ff.dv_temp, p)
        LaminartEqConv.fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, ff.dv_temp, p)
        #         LaminartEqConv.fun_dx_v1!(dx, ff.x, ff.C, z, p.x_V2, p)
        #         LaminartEqConv.fun_dy!(dy, y, ff.C, ff.x, ff.m, ff.dy_temp, p)
        LaminartEqConv.fun_x_equ!(ff.x, ff.C, z, ff.dy_temp, p.x_V2, p)
        LaminartEqConv.fun_y_equ!(y, ff.C, ff.x, ff.m, ff.dy_temp, p)
        LaminartEqConv.fun_dm!(dm, ff.m, ff.x, ff.dm_temp, p)
        LaminartEqConv.fun_dz!(dz, z, y, ff.H_z, ff.s, ff.dz_temp, p)
        LaminartEqConv.fun_ds!(ds, ff.s, ff.H_z, ff.ds_temp, p)

    end
    return nothing

end


"""
All arrays held in struct and temp arrays reused for different equations.
"""
mutable struct LamFunction_all_struct_reuse{T<:AbstractArray} <: Function

    x_lgn::T
    C::T
    H_z::T

    y::Any
    z::Any
    v_p::Any
    v_m::Any
    dx::Any
    dy::Any
    dm::Any
    dz::Any
    ds::Any
    dv_p::Any
    dv_m::Any

    tmp_a::T
    tmp_b::T

    tmp_A::T
    tmp_B::T
    tmp_C::T
end


function (ff::LamFunction_all_struct_reuse)(du, u, p, t)

    @inbounds begin


        ff.y = @view u[:, :, p.K+1:2*p.K, :]
        ff.z = @view u[:, :, 3*p.K+1:4*p.K, :]

        ff.v_p = @view u[:, :, 5*p.K+1:5*p.K+1, :]
        ff.v_m = @view u[:, :, 5*p.K+2:5*p.K+2, :]

        ff.dx = @view du[:, :, 1:p.K, :]
        ff.dy = @view du[:, :, p.K+1:2*p.K, :]
        ff.dm = @view du[:, :, 2*p.K+1:3*p.K, :]
        ff.dz = @view du[:, :, 3*p.K+1:4*p.K, :]
        ff.ds = @view du[:, :, 4*p.K+1:5*p.K, :]

        ff.dv_p = @view du[:, :, 5*p.K+1:5*p.K+1, :]
        ff.dv_m = @view du[:, :, 5*p.K+2:5*p.K+2, :]


        LaminartEqConv.fun_v_C!(
            ff.C,
            ff.v_p,
            ff.v_m,
            ff.tmp_a,
            ff.tmp_b,
            ff.tmp_A,
            ff.tmp_B,
            p,
        ) #a,b,C,D: buffers
        LaminartEqConv.fun_H_z!(ff.H_z, ff.z, ff.tmp_A, p) #A=buffer

        @. ff.tmp_A = @view u[:, :, 1:p.K, :] #x
        @. ff.tmp_B = @view u[:, :, 2*p.K+1:3*p.K, :] #m
        LaminartEqConv.fun_x_lgn!(ff.x_lgn, ff.tmp_A, p)

        LaminartEqConv.fun_dv!(ff.dv_p, ff.v_p, p.r, ff.x_lgn, ff.tmp_a, p) #a=buffer
        LaminartEqConv.fun_dv!(ff.dv_m, ff.v_m, .-p.r, ff.x_lgn, ff.tmp_a, p) #a=buff


        LaminartEqConv.fun_dx_v1!(ff.dx, ff.tmp_A, ff.C, ff.z, p.x_V2, p) #A:x
        LaminartEqConv.fun_dy!(
            ff.dy,
            ff.y,
            ff.C,
            ff.tmp_A,
            ff.tmp_B,
            ff.tmp_C,
            p,
        ) # A:x, B:m, C:buffer
        LaminartEqConv.fun_dm!(ff.dm, ff.tmp_B, ff.tmp_A, ff.tmp_C, p) # A:x, B:m, C:buffer

        @. ff.tmp_A = @view u[:, :, 4*p.K+1:5*p.K, :] #s
        LaminartEqConv.fun_dz!(ff.dz, ff.z, ff.y, ff.H_z, ff.tmp_A, ff.tmp_B, p) #A=s, B=buffer
        LaminartEqConv.fun_ds!(ff.ds, ff.tmp_A, ff.H_z, ff.tmp_B, p) #A=s, B=buffer


    end
    return nothing

end


"""
Uses equations with JuliaImage's imfilter for convolution on CPU.
"""
mutable struct LamFunction_imfil_cpu <: Function
    x_lgn::Any
    C::Any
    H_z::Any
    H_z_temp::Any
    v_C_temp1::Any
    v_C_temp2::Any
    v_C_tempA::Any
    W_temp::Any
end

function (ff::LamFunction_imfil_cpu)(du, u, p, t)
    @inbounds begin
        x = @view u[:, :, 1:p.K]
        y = @view u[:, :, p.K+1:2*p.K]
        m = @view u[:, :, 2*p.K+1:3*p.K]
        z = @view u[:, :, 3*p.K+1:4*p.K]
        s = @view u[:, :, 4*p.K+1:5*p.K]

        v_p = @view u[:, :, 5*p.K+1]
        v_m = @view u[:, :, 5*p.K+2]

        dx = @view du[:, :, 1:p.K]
        dy = @view du[:, :, p.K+1:2*p.K]
        dm = @view du[:, :, 2*p.K+1:3*p.K]
        dz = @view du[:, :, 3*p.K+1:4*p.K]
        ds = @view du[:, :, 4*p.K+1:5*p.K]

        dv_p = @view du[:, :, 5*p.K+1]
        dv_m = @view du[:, :, 5*p.K+2]


        LaminartEqImfilter.fun_x_lgn!(ff.x_lgn, x, p)
        LaminartEqImfilter.fun_v_C!(
            ff.C,
            v_p,
            v_m,
            ff.v_C_temp1,
            ff.v_C_temp2,
            ff.v_C_tempA,
            p,
        )
        LaminartEqImfilter.fun_H_z!(ff.H_z, z, ff.H_z_temp, p)

        LaminartEqImfilter.fun_dv!(dv_p, v_p, p.r, ff.x_lgn, p)
        LaminartEqImfilter.fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, p)
        LaminartEqImfilter.fun_dx_v1!(dx, x, ff.C, z, p.x_V2, p)
        LaminartEqImfilter.fun_dy!(dy, y, ff.C, x, m, ff.W_temp, p)
        LaminartEqImfilter.fun_dm!(dm, m, x, ff.W_temp, p)
        LaminartEqImfilter.fun_dz!(dz, z, y, ff.H_z, s, p)
        LaminartEqImfilter.fun_ds!(ds, s, ff.H_z, p)
    end
    return nothing
end



"""
Uses equations with JuliaImage's imfilter for convolution on GPU with IIR algorithom.
"""
mutable struct LamFunction_imfil_gpu_iir <: Function
    x_lgn::Any
    C::Any
    H_z::Any
    H_z_temp::Any
    v_C_temp1::Any
    v_C_temp2::Any
    v_C_tempA::Any
    W_temp::Any
end

function (ff::LamFunction_imfil_gpu_iir)(du, u, p, t)
    @inbounds begin
        x = @view u[:, :, 1:p.K]
        y = @view u[:, :, p.K+1:2*p.K]
        m = @view u[:, :, 2*p.K+1:3*p.K]
        z = @view u[:, :, 3*p.K+1:4*p.K]
        s = @view u[:, :, 4*p.K+1:5*p.K]

        v_p = @view u[:, :, 5*p.K+1]
        v_m = @view u[:, :, 5*p.K+2]

        dx = @view du[:, :, 1:p.K]
        dy = @view du[:, :, p.K+1:2*p.K]
        dm = @view du[:, :, 2*p.K+1:3*p.K]
        dz = @view du[:, :, 3*p.K+1:4*p.K]
        ds = @view du[:, :, 4*p.K+1:5*p.K]

        dv_p = @view du[:, :, 5*p.K+1]
        dv_m = @view du[:, :, 5*p.K+2]


        LaminartEqImfilterGPU_IIR.fun_x_lgn!(ff.x_lgn, x, p)
        LaminartEqImfilterGPU_IIR.fun_v_C!(
            ff.C,
            v_p,
            v_m,
            ff.v_C_temp1,
            ff.v_C_temp2,
            ff.v_C_tempA,
            p,
        )
        LaminartEqImfilterGPU_IIR.fun_H_z!(ff.H_z, z, ff.H_z_temp, p)

        LaminartEqImfilterGPU_IIR.fun_dv!(dv_p, v_p, p.r, ff.x_lgn, p)
        LaminartEqImfilterGPU_IIR.fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, p)
        LaminartEqImfilterGPU_IIR.fun_dx_v1!(dx, x, ff.C, z, p.x_V2, p)
        LaminartEqImfilterGPU_IIR.fun_dy!(dy, y, ff.C, x, m, ff.W_temp, p)
        LaminartEqImfilterGPU_IIR.fun_dm!(dm, m, x, ff.W_temp, p)
        LaminartEqImfilterGPU_IIR.fun_dz!(dz, z, y, ff.H_z, s, p)
        LaminartEqImfilterGPU_IIR.fun_ds!(ds, s, ff.H_z, p)
    end
    return nothing
end


"""
Uses equations with JuliaImage's imfilter for convolution on GPU with FIR algorithom.
"""
mutable struct LamFunction_imfil_gpu_fir <: Function
    x_lgn::Any
    C::Any
    H_z::Any
    H_z_temp::Any
    v_C_temp1::Any
    v_C_temp2::Any
    v_C_tempA::Any
    W_temp::Any
end

function (ff::LamFunction_imfil_gpu_fir)(du, u, p, t)
    @inbounds begin
        x = @view u[:, :, 1:p.K]
        y = @view u[:, :, p.K+1:2*p.K]
        m = @view u[:, :, 2*p.K+1:3*p.K]
        z = @view u[:, :, 3*p.K+1:4*p.K]
        s = @view u[:, :, 4*p.K+1:5*p.K]

        v_p = @view u[:, :, 5*p.K+1]
        v_m = @view u[:, :, 5*p.K+2]

        dx = @view du[:, :, 1:p.K]
        dy = @view du[:, :, p.K+1:2*p.K]
        dm = @view du[:, :, 2*p.K+1:3*p.K]
        dz = @view du[:, :, 3*p.K+1:4*p.K]
        ds = @view du[:, :, 4*p.K+1:5*p.K]

        dv_p = @view du[:, :, 5*p.K+1]
        dv_m = @view du[:, :, 5*p.K+2]


        LaminartEqImfilterGPU_FIR.fun_x_lgn!(ff.x_lgn, x, p)
        LaminartEqImfilterGPU_FIR.fun_v_C!(
            ff.C,
            v_p,
            v_m,
            ff.v_C_temp1,
            ff.v_C_temp2,
            ff.v_C_tempA,
            p,
        )
        LaminartEqImfilterGPU_FIR.fun_H_z!(ff.H_z, z, ff.H_z_temp, p)

        LaminartEqImfilterGPU_FIR.fun_dv!(dv_p, v_p, p.r, ff.x_lgn, p)
        LaminartEqImfilterGPU_FIR.fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, p)
        LaminartEqImfilterGPU_FIR.fun_dx_v1!(dx, x, ff.C, z, p.x_V2, p)
        LaminartEqImfilterGPU_FIR.fun_dy!(dy, y, ff.C, x, m, ff.W_temp, p)
        LaminartEqImfilterGPU_FIR.fun_dm!(dm, m, x, ff.W_temp, p)
        LaminartEqImfilterGPU_FIR.fun_dz!(dz, z, y, ff.H_z, s, p)
        LaminartEqImfilterGPU_FIR.fun_ds!(ds, s, ff.H_z, p)
    end
    return nothing
end


"""
Uses equations with JuliaImage's imfilter for convolution on GPU with FFT algorithom.
"""
mutable struct LamFunction_imfil_gpu_fft <: Function
    x_lgn::Any
    C::Any
    H_z::Any
    H_z_temp::Any
    v_C_temp1::Any
    v_C_temp2::Any
    v_C_tempA::Any
    W_temp::Any
end

function (ff::LamFunction_imfil_gpu_fft)(du, u, p, t)
    @inbounds begin
        x = @view u[:, :, 1:p.K]
        y = @view u[:, :, p.K+1:2*p.K]
        m = @view u[:, :, 2*p.K+1:3*p.K]
        z = @view u[:, :, 3*p.K+1:4*p.K]
        s = @view u[:, :, 4*p.K+1:5*p.K]

        v_p = @view u[:, :, 5*p.K+1]
        v_m = @view u[:, :, 5*p.K+2]

        dx = @view du[:, :, 1:p.K]
        dy = @view du[:, :, p.K+1:2*p.K]
        dm = @view du[:, :, 2*p.K+1:3*p.K]
        dz = @view du[:, :, 3*p.K+1:4*p.K]
        ds = @view du[:, :, 4*p.K+1:5*p.K]

        dv_p = @view du[:, :, 5*p.K+1]
        dv_m = @view du[:, :, 5*p.K+2]


        LaminartEqImfilterGPU_FFT.fun_x_lgn!(ff.x_lgn, x, p)
        LaminartEqImfilterGPU_FFT.fun_v_C!(
            ff.C,
            v_p,
            v_m,
            ff.v_C_temp1,
            ff.v_C_temp2,
            ff.v_C_tempA,
            p,
        )
        LaminartEqImfilterGPU_FFT.fun_H_z!(ff.H_z, z, ff.H_z_temp, p)

        LaminartEqImfilterGPU_FFT.fun_dv!(dv_p, v_p, p.r, ff.x_lgn, p)
        LaminartEqImfilterGPU_FFT.fun_dv!(dv_m, v_m, .-p.r, ff.x_lgn, p)
        LaminartEqImfilterGPU_FFT.fun_dx_v1!(dx, x, ff.C, z, p.x_V2, p)
        LaminartEqImfilterGPU_FFT.fun_dy!(dy, y, ff.C, x, m, ff.W_temp, p)
        LaminartEqImfilterGPU_FFT.fun_dm!(dm, m, x, ff.W_temp, p)
        LaminartEqImfilterGPU_FFT.fun_dz!(dz, z, y, ff.H_z, s, p)
        LaminartEqImfilterGPU_FFT.fun_ds!(ds, s, ff.H_z, p)
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
# 	A_temp::AbstractArray
# 	B_temp::AbstractArray
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
#         fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.V_temp_1, ff.V_temp_2, ff.A_temp, ff.B_temp, p)
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
# 	A_temp::T
# 	B_temp::T
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
# #         fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.V_temp_1, ff.V_temp_2, ff.A_temp, ff.B_temp, p)
# #         fun_H_z!(ff.H_z, ff.z, ff.H_z_temp, p)

# #         fun_dv!(dv_p, ff.v_p, p.r, ff.x_lgn, ff.dv_temp, p)
# #         fun_dv!(dv_m, ff.v_m, .-p.r, ff.x_lgn, ff.dv_temp, p)
# #         fun_dx_v1!(dx, ff.x, ff.C, ff.z, p.x_V2, p)
# #         fun_dy!(dy, ff.y, ff.C, ff.x, ff.m, ff.dy_temp, p)
# #         fun_dm!(dm, ff.m, ff.x, ff.dm_temp, p)
# #         fun_dz!(dz, ff.z, ff.y, ff.H_z, ff.s, ff.dz_temp, p)
# #         fun_ds!(ds, ff.s, ff.H_z, ff.ds_temp, p)

# # 		fun_x_lgn!(ff.x_lgn, x, p)
#         fun_v_C!(ff.C, v_p, v_m, ff.V_temp_1, ff.V_temp_2, ff.A_temp, ff.B_temp, p)
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
# 	A_temp::T
# 	B_temp::T
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
# #         fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.V_temp_1, ff.V_temp_2, ff.A_temp, ff.B_temp, p)
# #         fun_H_z!(ff.H_z, ff.z, ff.H_z_temp, p)

# #         fun_dv!(dv_p, ff.v_p, p.r, ff.x_lgn, ff.dv_temp, p)
# #         fun_dv!(dv_m, ff.v_m, .-p.r, ff.x_lgn, ff.dv_temp, p)
# #         fun_dx_v1!(dx, ff.x, ff.C, ff.z, p.x_V2, p)
# #         fun_dy!(dy, ff.y, ff.C, ff.x, ff.m, ff.dy_temp, p)
# #         fun_dm!(dm, ff.m, ff.x, ff.dm_temp, p)
# #         fun_dz!(dz, ff.z, ff.y, ff.H_z, ff.s, ff.dz_temp, p)
# #         fun_ds!(ds, ff.s, ff.H_z, ff.ds_temp, p)

# # 		fun_x_lgn!(ff.x_lgn, x, p)
#         fun_v_C!(ff.C, v_p, v_m, ff.V_temp_1, ff.V_temp_2, ff.A_temp, ff.B_temp, p)
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
# 	A_temp::T
# 	B_temp::T
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
# 	A_temp::T
# 	B_temp::T
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
# 	A_temp::AbstractArray
# 	B_temp::AbstractArray
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
#         fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.V_temp_1, ff.V_temp_2, ff.A_temp, ff.B_temp, p)
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
#         fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.V_temp_1, ff.V_temp_2, ff.A_temp, ff.B_temp, p)
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
# 	A_temp::CuArray{Float32,4,Nothing}
# 	B_temp::CuArray{Float32,4,Nothing}
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
# 	A_temp
# 	B_temp
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
# 	A_temp::CuArray{Float32,4,Nothing}
# 	B_temp::CuArray{Float32,4,Nothing}
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
# # 		ff.A_temp_ = @view ff.A_temp[:,:,:,:]
# # 		ff.B_temp_ = @view ff.B_temp[:,:,:,:]

#         fun_x_lgn!(ff.x_lgn, ff.x, p)
#         fun_v_C!(ff.C, ff.v_p, ff.v_m, ff.V_temp_1, ff.V_temp_2, ff.A_temp, ff.B_temp, p)
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


end
