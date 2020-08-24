using MEngProject
using Test
using Images
img = convert(Array{Float32,2}, load(datadir("img", "kan_sq_cont_l.png"))
p = Laminart.kernels(img, Parameters.parameters)
p = Laminart.add_I_u_p(img, p)

@testset "MEngProject.jl" begin

    out = zeros(p.dim_i, p.dim_j)
    b = reshape(ones(p.dim_i, p.dim_j * p.K), p.dim_i, p.dim_j, p.K)
    c = 2 .* ones(p.dim_i, p.dim_j)


    @test LaminartEqConv.conv!()≈
    @test LaminartEqConv.fun_f!()
    @test LaminartEqConv.I_u!()
    @test LaminartEqConv.fun_dv!()
    @test LaminartEqConv.fun_x_lgn()
    @test LaminartEqConv.fun_v_C!()
    @test LaminartEqConv.fun_dx_v1!()
    @test LaminartEqConv.fun_dy!()
    @test LaminartEqConv.fun_dm!()
    @test LaminartEqConv.fun_dz!()
    @test LaminartEqConv.fun_ds!()
    @test LaminartEqConv.fun_H_z!()
    @test LaminartEqConv.fun_dx_v2()
    @test LaminartEqConv.fun_dy_v2!()
    @test LaminartEqConv.fun_x_equ!()
    @test LaminartEqConv.fun_y_equ!()


    @test LaminartEqImfilter.conv!()
    @test LaminartEqImfilter.fun_f!()
    @test LaminartEqImfilter.I_u!()
    @test LaminartEqImfilter.fun_dv!()
    @test LaminartEqImfilter.fun_x_lgn()
    @test LaminartEqImfilter.fun_v_C!()
    @test LaminartEqImfilter.fun_dx_v1!()
    @test LaminartEqImfilter.fun_dy!()
    @test LaminartEqImfilter.fun_dm!()
    @test LaminartEqImfilter.fun_dz!()
    @test LaminartEqImfilter.fun_ds!()
    @test LaminartEqImfilter.fun_H_z!()
    @test LaminartEqImfilter.fun_dx_v2()
    @test LaminartEqImfilter.fun_dy_v2!()
    @test LaminartEqImfilter.fun_x_equ!()
    @test LaminartEqImfilter.fun_y_equ!()


    @test LaminartEqImfilterGPU_FFT.conv!()
    @test LaminartEqImfilterGPU_FFT.fun_f!()
    @test LaminartEqImfilterGPU_FFT.I_u!()
    @test LaminartEqImfilterGPU_FFT.fun_dv!()
    @test LaminartEqImfilterGPU_FFT.fun_x_lgn()
    @test LaminartEqImfilterGPU_FFT.fun_v_C!()
    @test LaminartEqImfilterGPU_FFT.fun_dx_v1!()
    @test LaminartEqImfilterGPU_FFT.fun_dy!()
    @test LaminartEqImfilterGPU_FFT.fun_dm!()
    @test LaminartEqImfilterGPU_FFT.fun_dz!()
    @test LaminartEqImfilterGPU_FFT.fun_ds!()
    @test LaminartEqImfilterGPU_FFT.fun_H_z!()
    @test LaminartEqImfilterGPU_FFT.fun_dx_v2()
    @test LaminartEqImfilterGPU_FFT.fun_dy_v2!()
    @test LaminartEqImfilterGPU_FFT.fun_x_equ!()
    @test LaminartEqImfilterGPU_FFT.fun_y_equ!()


    @test LaminartEqImfilterGPU_FIR.conv!()
    @test LaminartEqImfilterGPU_FIR.fun_f!()
    @test LaminartEqImfilterGPU_FIR.I_u!()
    @test LaminartEqImfilterGPU_FIR.fun_dv!()
    @test LaminartEqImfilterGPU_FIR.fun_x_lgn()
    @test LaminartEqImfilterGPU_FIR.fun_v_C!()
    @test LaminartEqImfilterGPU_FIR.fun_dx_v1!()
    @test LaminartEqImfilterGPU_FIR.fun_dy!()
    @test LaminartEqImfilterGPU_FIR.fun_dm!()
    @test LaminartEqImfilterGPU_FIR.fun_dz!()
    @test LaminartEqImfilterGPU_FIR.fun_ds!()
    @test LaminartEqImfilterGPU_FIR.fun_H_z!()
    @test LaminartEqImfilterGPU_FIR.fun_dx_v2()
    @test LaminartEqImfilterGPU_FIR.fun_dy_v2!()
    @test LaminartEqImfilterGPU_FIR.fun_x_equ!()
    @test LaminartEqImfilterGPU_FIR.fun_y_equ!()


    @test LaminartEqImfilterGPU_IIR.conv!()
    @test LaminartEqImfilterGPU_IIR.fun_f!()
    @test LaminartEqImfilterGPU_IIR.I_u!()
    @test LaminartEqImfilterGPU_IIR.fun_dv!()
    @test LaminartEqImfilterGPU_IIR.fun_x_lgn()
    @test LaminartEqImfilterGPU_IIR.fun_v_C!()
    @test LaminartEqImfilterGPU_IIR.fun_dx_v1!()
    @test LaminartEqImfilterGPU_IIR.fun_dy!()
    @test LaminartEqImfilterGPU_IIR.fun_dm!()
    @test LaminartEqImfilterGPU_IIR.fun_dz!()
    @test LaminartEqImfilterGPU_IIR.fun_ds!()
    @test LaminartEqImfilterGPU_IIR.fun_H_z!()
    @test LaminartEqImfilterGPU_IIR.fun_dx_v2()
    @test LaminartEqImfilterGPU_IIR.fun_dy_v2!()
    @test LaminartEqImfilterGPU_IIR.fun_x_equ!()
    @test LaminartEqImfilterGPU_IIR.fun_y_equ!()


    @test LaminartFunc.LamFunction()
    @test LaminartFunc.LamFunction_allStruct()
    @test LaminartFunc.LamFunction_gpu_reuse()
    @test LaminartFunc.LamFunction_equ()
    @test LaminartFunc.LamFunction_all_struct_reuse()
    @test LaminartFunc.LamFunction_imfil_cpu()
    @test LaminartFunc.LamFunction_imfil_gpu_iir()
    @test LaminartFunc.LamFunction_imfil_gpu_fir()
    @test LaminartFunc.LamFunction_imfil_gpu_fft()


    @test LaminartInitFunc.parameterInit_conv_gpu()
    @test LaminartInitFunc.parameterInit_conv_gpu_noise()
    @test LaminartInitFunc.parameterInit_conv_cpu()
    @test LaminartInitFunc.parameterInit_imfil_cpu()

    @test LaminartInitFunc.reshape2d_4d()
    @test LaminartInitFunc.add_I_u_p()

    @test LaminartInitFunc.kernels_conv_gpu()
    @test LaminartInitFunc.kernels_conv_cpu()
    @test LaminartInitFunc.kernels_imfil_cpu()


    @test kern_d_ph()
    @test kern_d_pv()
    @test kern_d_mh()
    @test kern_d_mv()
    @test kern_d_p()
    @test kern_d_m()
    @test kern_A()
    @test kern_B()
    @test gaussian_rot()
    @test fun_R()

    @test Parameters.para_var()
    @test Parameters.para_var_k()
    @test Parameters.parameters_f32
    @test Parameters.parameters_f64
end
