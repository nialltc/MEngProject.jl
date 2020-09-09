"""
# script test_illusions

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-08-20


Script to run set of images with LAMINART and plot layers and activation.
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

batch = 9003

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
    "mo05709.png",
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

batch_ = string(batch, "_", rand(1000:9999));
mkdir(plotsdir(string("input_plot", batch_)));

tspan = (0.0f0, 800f0);

for file in files
    p = LaminartInitFunc.parameterInit_conv_cpu(
        datadir("img", file),
        Parameters.parameters_f32,
    )

    Utils.plot_gs(
        p.I[:, :, 1, 1],
        name = file,
        loc = plotsdir(string("input_plot", batch_)),
        clbar = true,
        axMax = 1,
        save = true,
    )
end
