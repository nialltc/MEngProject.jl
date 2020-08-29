"""
# script test_illusions

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-08-20


Script to run set of images with LAMINART and plot layers and activation.
# Examples

```jldoctest
julia>
```
"""

using DrWatson
@quickactivate "MEngProject"
using MEngProject,
    CUDA,
    DifferentialEquations,
    PyPlot,
    NNlib,
    ImageFiltering,
    Images,
    ImageIO,
    MEngProject.LaminartKernels,
    MEngProject.LaminartInitFunc,
    MEngProject.Utils,
    BenchmarkTools,
    Test

using OrdinaryDiffEq,
    ParameterizedFunctions, LSODA, Sundials, DiffEqDevTools, Noise


batch = 1002


# files = readdir(datadir("img"))

files = [
        	"Iine_gap_1_100_gs.png",
         "Iine_gap_2_100_gs.png",
         "Iine_gap_3_100_gs.png",
         "Iine_gap_4_100_gs.png",
         "Iines_gaps_100_gs.png",
         "diag_dots_100_gs.png",
         "diag_gap_100_gs.png",
         "kan_sq_cont.png",
         "kan_sq_cont_l.png",
    #      "mo05709.png",
    "stairs_100gs.png",
    "stairs_200gs.png",
    "viper00187.png",
    "viper00661.jpg",
    "viper00715.png",
    "viper00717.png",
    "viper00720.png",
    "viper00721.png",
    "viper00842.jpg",
    "viper00891.jpg",
    "viper00904.jpg",
    "viper00921.jpg",
    "viper01006.jpg",
    "viper01333.jpg",
]

batch_ = string(batch, "_", rand(1000:9999))
mkdir(plotsdir(string("illusions", batch_)))
mkdir(plotsdir(string("illusions_nocb", batch_)))

tspan = (0.0f0, 800f0)

for file in files
    try
        p = LaminartInitFunc.parameterInit_conv_gpu(
            datadir("img", file),
            Parameters.parameters_f32,
        )

        u0 = cu(reshape(
            zeros(Float32, p.dim_i, p.dim_j * (5 * p.K + 2)),
            p.dim_i,
            p.dim_j,
            5 * p.K + 2,
            1,
        ))

        arr1 = similar(@view u0[:, :, 1:2, :])
        arr2 = similar(@view u0[:, :, 1:1, :])

        f = LaminartFunc.LamFunction(
            arr1, #x
            similar(arr1), #m
            similar(arr1), #s
            arr2, #x_lgn,
            similar(arr1), #C,
            similar(arr1), #H_z,
            similar(arr1), # dy_temp,
            similar(arr1), # dm_temp,
            similar(arr1), # dz_temp,
            similar(arr1), # ds_temp,
            similar(arr2), # dv_temp,
            similar(arr1), # H_z_temp,
            similar(arr2), #  V_temp_1,
            similar(arr2), #  V_temp_2,
            similar(arr1), #  A_temp,
            similar(arr1), #   B_temp
        )

        prob = ODEProblem(f, u0, tspan, p)
        # 	@benchmark sol = solve(prob)
        sol = solve(prob)

        # plots
        for t ∈ [25, 50, 100, 200, 400, 800]
            Utils.plot_k2_vsep1(sol, t, "illusions", batch_, file)
            Utils.plot_k2_vsep1(sol, t, "illusions_nocb", batch_, file, cb=false)
        end


        Utils.plot_t_act(sol, "illusions", batch_, file)
    catch err
        print(err)
    finally
        u0 = nothing
        p = nothing
        arr1 = nothing
        arr2 = nothing
        f = nothing
        prob = nothing
        sol = nothing
        CUDA.reclaim()
    end
end
