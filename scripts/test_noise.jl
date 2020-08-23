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

batch = 4


# files = readdir(datadir("img"))
files = ["stairs_100gs.png"]
# files = ["kan_sq_cont_l.png", "stairs_100gs.png"]

tspan = (0.0f0, 800f0)

batch_ = string(batch, "_", rand(1000:9999))
mkdir(plotsdir(string("noise", batch_)))
for file in files[1:end]
    for noise in [0.01f0, 0.05f0, 0.1f0, 0.2f0, 0.5f0, 1f0,]
		img = convert(Array{Float32,2},  load(datadir("img",files[1])));
		img = mult_gauss(img,noise)
        fig, ax = plt.subplots()
		im = ax.imshow(
			img,
			cmap = matplotlib.cm.gray,
			vmax = 1,
			vmin = 0,
		)
		plt.title("Input image, noise \$σ=$noise\$")
		plt.axis("off")
		fig.tight_layout()
		plt.savefig(plotsdir(
			string("noise", batch_),
			string(
				file,
				"_input_noise_",
				noise,
				".png",
			),
		))
		close("all")
		img = nothing

        try
            p = LaminartInitFunc.parameterInit_conv_gpu_noise(
                datadir("img", file),
                Parameters.parameters_f32,
                noise,
            )

            u0 = cu(reshape(
                zeros(Float32, p.dim_i, p.dim_j * (5 * p.K + 2)),
                p.dim_i,
                p.dim_j,
                5 * p.K + 2,
                1,
            ))

            arr1 = similar(u0[:, :, 1:2, :])
            arr2 = similar(u0[:, :, 1:1, :])

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
                        k2 = k + 1
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
                        cbar.ax.set_xlabel("\$k=$k2\$")
                        cbar = fig.colorbar(im, shrink = 0.9, ax = ax)
                        cbar.ax.set_xlabel("\$k=$k\$")
                        layer = Utils.layers[k]
                        plt.title("Layer: $layer, \$t=$t\$, noise \$σ=$noise\$")
                        plt.axis("off")
                        fig.tight_layout()
                        plt.savefig(plotsdir(
                            string("noise", batch_),
                            string(
                                file,
                                "_noise_",
                                noise,
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
                    cbar.ax.set_xlabel("\$v^+\$")

                    layer = Utils.layers[k]
                    plt.title("Layer: $layer, \$t=$t\$, noise \$σ=$noise\$")
                    plt.axis("off")
                    fig.tight_layout()

                    plt.savefig(plotsdir(
                        string("noise", batch_),
                        string(
                            file,
                            "_noise_",
                            noise,
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
        finally
            nothing
        end


        # time plot
        fig, axs = plt.subplots()

        for k ∈ 1:12
            v3 = @view sol[:, :, k, 1, end]
            v4 = @view sol[findmax(v3)[2][1], findmax(v3)[2][2], k, 1, :]
            layer = Utils.layers_1[k]
            axs.plot(v4, Utils.lines[k], label = "$layer")
        end
        axs.set_xlabel("Time")
        axs.set_ylabel("Activation")
        plt.title("Noise \$σ=$noise\$")
        plt.legend()
        fig.tight_layout()
        plt.savefig(plotsdir(
            string("noise", batch_),
            string(file, "_noise_", noise, "_time.png"),
        ))
        u0 = nothing
        p = nothing
        arr1 = nothing
        arr2 = nothing
        f = nothing
        prob = nothing
        sol = nothing
        close("all")
    end
end
