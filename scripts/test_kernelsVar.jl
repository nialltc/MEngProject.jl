"""
# script test_kernelsVar

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-08-20


Script to vary kernel parameters of LAMINART and plot layers and activation.
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
    MEngProject.LaminartKernels,
    MEngProject.LaminartInitFunc,
    MEngProject.Utils,
    BenchmarkTools,
    Test

using OrdinaryDiffEq,
    ParameterizedFunctions, LSODA, Sundials, DiffEqDevTools, Noise

batch = 1000


# files = readdir(datadir("img"))
files = ["kan_sq_cont_l.png"]


tspan = (0.0f0, 800f0)

batch_ = string(batch, "_", rand(1000:9999))
mkdir(plotsdir(string("kernVar", batch_)))

para_sets = [
    (
        H_fact = 13.0f0,
        T_fact = [0.87f0, 0.13f0],
        T_p_m = 0.302f0,
        W_p_same_fact = 39f0,
        W_m_same_fact = 330f0,
        H_l = 19,
        W_l = 19,
    ),
    (
        H_fact = 26.0f0,
        T_fact = [0.87f0, 0.13f0],
        T_p_m = 0.302f0,
        W_p_same_fact = 39f0,
        W_m_same_fact = 330f0,
        H_l = 19,
        W_l = 19,
    ),
    (
        H_fact = 7.0f0,
        T_fact = [0.87f0, 0.13f0],
        T_p_m = 0.302f0,
        W_p_same_fact = 39f0,
        W_m_same_fact = 330f0,
        H_l = 19,
        W_l = 19,
    ),
    (
        H_fact = 13.0f0,
        T_fact = [1.0f0, 0.0f0],
        T_p_m = 0.302f0,
        W_p_same_fact = 39f0,
        W_m_same_fact = 330f0,
        H_l = 19,
        W_l = 19,
    ),
    (
        H_fact = 13.0f0,
        T_fact = [0.5f0, 0.5f0],
        T_p_m = 0.302f0,
        W_p_same_fact = 39f0,
        W_m_same_fact = 330f0,
        H_l = 19,
        W_l = 19,
    ),
    (
        H_fact = 13.0f0,
        T_fact = [0.87f0, 0.13f0],
        T_p_m = 0.15f0,
        W_p_same_fact = 39f0,
        W_m_same_fact = 330f0,
        H_l = 19,
        W_l = 19,
    ),
    (
        H_fact = 13.0f0,
        T_fact = [0.87f0, 0.13f0],
        T_p_m = 0.6f0,
        W_p_same_fact = 39f0,
        W_m_same_fact = 330f0,
        H_l = 19,
        W_l = 19,
    ),
    (
        H_fact = 13.0f0,
        T_fact = [0.87f0, 0.13f0],
        T_p_m = 0.302f0,
        W_p_same_fact = 80f0,
        W_m_same_fact = 330f0,
        H_l = 19,
        W_l = 19,
    ),
    (
        H_fact = 13.0f0,
        T_fact = [0.87f0, 0.13f0],
        T_p_m = 0.302f0,
        W_p_same_fact = 20f0,
        W_m_same_fact = 330f0,
        H_l = 19,
        W_l = 19,
    ),
    (
        H_fact = 13.0f0,
        T_fact = [0.87f0, 0.13f0],
        T_p_m = 0.302f0,
        W_p_same_fact = 39f0,
        W_m_same_fact = 400f0,
        H_l = 19,
        W_l = 19,
    ),
    (
        H_fact = 13.0f0,
        T_fact = [0.87f0, 0.13f0],
        T_p_m = 0.302f0,
        W_p_same_fact = 39f0,
        W_m_same_fact = 200f0,
        H_l = 19,
        W_l = 19,
    ),
]
test_name = [
    "base",
    "H_double",
    "H_half",
    "T1_0",
    "T05_05",
    "Tm_half",
    "Tm_double",
    "Wp_up",
    "Wp_half",
    "Wm_up",
    "Wm_down",
]

test_name_plt = [
    "Base",
    "\$H\$ doubled",
    "\$H\$ halfed",
    "\$T=[1,0]\$ ",
    "\$T=[0.5,0.5]\$",
    "\$T^-\$ halved",
    "\$T^+\$ doubled",
    "\$W^+\$ increased",
    "\$W^+\$ halved",
    "\$W^-\$ increased",
    "\$W^-\$ decreased",
]

for file in files
    for para_test in enumerate(para_sets)
        try
            p = LaminartInitFunc.parameterInit_conv_gpu(
                datadir("img", file),
                Parameters.para_var_k(para_test[2]),
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


            for t ∈ [25, 50, 100, 200, 400, 800]
                # 				for t ∈ [25,50,100]
                @inbounds begin
                    v0 = @view sol(t)[:, :, :, 1]
                    axMax = findmax(v0)[1]

                    for k ∈ 1:2:10
                        fig, ax = plt.subplots()

                        v1 = @view sol(t)[:, :, k, 1]
                        v2 = @view sol(t)[:, :, k+1, 1]
                        im = ax.imshow(
                            v1,
                            cmap = matplotlib.cm.PRGn,
                            vmax = axMax,
                            vmin = -axMax,
                        )
                        im2 = ax.imshow(
                            v2,
                            cmap = matplotlib.cm.RdBu_r,
                            vmax = axMax,
                            vmin = -axMax,
                            alpha = 0.5,
                        )

                        cbar = fig.colorbar(im2, shrink = 0.9, ax = ax)
                        cbar.ax.set_xlabel("\$k=2\$")
                        cbar = fig.colorbar(im, shrink = 0.9, ax = ax)
                        cbar.set_alpha(0.5)
                        cbar.draw_all()
                        cbar.ax.set_xlabel("\$k=1\$")
                        layer = Utils.layers[k]
                        plt.title(string(
                            "Layer: $layer, \$t=$t\$, ",
                            test_name_plt[para_test[1]],
                        ))
                        plt.axis("off")
                        fig.tight_layout()
                        plt.savefig(plotsdir(
                            string("kernVar", batch_),
                            string(
                                file,
                                "_para_",
                                test_name[para_test[1]],
                                "_t",
                                t,
                                "_",
                                Utils.la[k],
                                ".png",
                            ),
                        ))
                        close("all")
                    end


                    k = 11
                    fig, ax = plt.subplots()
                    v1 = @view sol[:, :, k, 1, t]
                    v2 = @view sol[:, :, k+1, 1, t]
                    im = ax.imshow(
                        v1,
                        cmap = matplotlib.cm.PRGn,
                        vmax = axMax,
                        vmin = -axMax,
                    )
                    im2 = ax.imshow(
                        v2,
                        cmap = matplotlib.cm.RdBu_r,
                        vmax = axMax,
                        vmin = -axMax,
                        alpha = 0.5,
                    )

                    cbar = fig.colorbar(im2, shrink = 0.9, ax = ax)
                    cbar.ax.set_xlabel("\$v^-\$")
                    cbar = fig.colorbar(im, shrink = 0.9, ax = ax)
                    cbar.set_alpha(0.5)
                    cbar.draw_all()
                    cbar.ax.set_xlabel("\$v^+\$")

                    layer = Utils.layers[k]
                    plt.title(string(
                        "Layer: $layer, \$t=$t\$, ",
                        test_name_plt[para_test[1]],
                    ))
                    plt.axis("off")
                    fig.tight_layout()

                    plt.savefig(plotsdir(
                        string("kernVar", batch_),
                        string(
                            file,
                            "_para_",
                            test_name[para_test[1]],
                            "_t",
                            t,
                            "_",
                            Utils.la[k],
                            ".png",
                        ),
                    ))
                    close("all")
                end
            end

            @inbounds begin
                # time plot
                fig, axs = plt.subplots()

                for k ∈ 1:12
                    v3 = @view sol[:, :, k, 1, end]
                    v4 =
                        @view sol[findmax(v3)[2][1], findmax(v3)[2][2], k, 1, :]
                    layer = Utils.layers_1[k]
                    axs.plot(sol.t, v4, Utils.lines[k], label = "$layer", alpha=0.8)
                end
                axs.set_xlabel("Time")
                axs.set_ylabel("Activation")
                plt.title(test_name_plt[para_test[1]])
                plt.legend()
                fig.tight_layout()
                plt.savefig(plotsdir(
                    string("kernVar", batch_),
                    string(
                        file,
                        "_para_",
                        test_name[para_test[1]],
                        "_time.png",
                    ),
                ))
                u0 = nothing
                p = nothing
                arr1 = nothing
                arr2 = nothing
                f = nothing
                prob = nothing
                sol = nothing
                close("all")
				CUDA.reclaim()
            end
		catch err
            print(err)
        end
    end
end
