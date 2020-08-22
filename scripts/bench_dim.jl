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

batch = 1


files = readdir(datadir("res_test"))
# files = ["kan_sq_cont_l.png"]
let
@inbounds begin
    tspan = (0.0f0, 10f0)

    batch_ = string(batch, "_", rand(1000:9999))
    mkdir(plotsdir(string("bench_dim", batch_)))


    test_name = ["025", "050", "075", "100", "200", "300", "400"]
    test_name_plt = [
        "\$25×25\$",
        "\$50×50\$",
        "\$75×75\$",
        "\$100×100\$",
        "\$200×200\$",
        "\$300×300\$",
        "\$400×400\$",
    ]
	test_no = 0
    benchm_gpu = []
	benchm_cpu = []
	y1Res_gpu = []
	y1Res_cpu = []

    for file in files[1:1]


        # 		for para_test in para_sets
        test_no += 1
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
        append!(benchm_gpu, @benchmark solve(prob))
        sol = solve(prob)


        t = 10
        v0 = @view sol(t)[:, :, :, 1]
        axMax = findmax(v0)[1]


        k = 7
		k2 = 8
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
        plt.title(string(
            "Layer: $layer, \$t=$t\$, resolution=",
            test_name_plt[test_no],
        ))
        plt.axis("off")
        fig.tight_layout()
        plt.savefig(plotsdir(
            string("bench_dim", batch_),
            string(
                file,
                "_res_",
                test_name[test_no],
                "_t",
                t,
                "_",
                Utils.la[k],
                ".png",
            ),
        ))
        close("all")

		v3 = @view sol[:,:,7:7,:,:]
		append!(y1Res, Array(v3))
		u0 = nothing
		p = nothing
		arr1 = nothing
		arr2 = nothing
		f = nothing
		prob = nothing
		sol = nothing


		p = LaminartInitFunc.parameterInit_conv_cpu(
            datadir("img", file),
            Parameters.parameters_f32,
        )

        u0 = reshape(
            zeros(Float32, p.dim_i, p.dim_j * (5 * p.K + 2)),
            p.dim_i,
            p.dim_j,
            5 * p.K + 2,
            1,
        )

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
        append!(benchm_cpu, @benchmark solve(prob))
        sol = solve(prob)

		append!(y1Res_cpu, sol[:,:,7:7,:,:])
		u0 = nothing
		p = nothing
		arr1 = nothing
		arr2 = nothing
		f = nothing
		prob = nothing
		sol = nothing
	end

    # time plot
    fig, axs = plt.subplots()

    for result ∈ enumerate(y1Res_gpu)
		lab = "$test_name[result[1]]"
        axs.plot(result[2][findmax(result[2][:, :, 1, 1, end])[2][1], findmax(result[2][:, :, 1, 1, end])[2][2], k, 1, :], c = Utils.Colour[result[1]], "--", label = "$lab GPU")
    end

	for result ∈ enumerate(y1Res_cpu)
		lab = "$test_name[result[1]]"
		axs.plot(result[2][findmax(result[2][:, :, 1, 1, end])[2][1], findmax(result[2][:, :, 1, 1, end])[2][2], k, 1, :], c = Utils.Colour[result[1]],":", label = "$lab CPU")
	end
    axs.set_xlabel("Time")
    axs.set_ylabel("Activation")
    plt.title("L2/3, \$k=1\$")
    plt.legend()
    fig.tight_layout()
    plt.savefig(plotsdir(
        string("bench_dim", batch_),
        string(file, "_para_", test_name[test_no], "_time.png"),
    ))
    close("all")



    # benchmark plot

    fig, ax = plt.subplots()
    for bm ∈ enumerate(benchm_gpu)
        ax.scatter(
		median(bm[2].times) * 1e-9,
		test_name_plt[bm[1]],
            label = "GPU",
			color=Utils.colours[1],
            alpha = 0.3,
            edgecolors = "none",
        )
    end

	for bm ∈ enumerate(benchm_cpu)
		ax.scatter(
            median(bm[2].times) * 1e-9,
			test_name_plt[bm[1]],
			label = "CPU",
			alpha = 0.3,
			color=Utils.colours[2],
			edgecolors = "none",
		)
	end

    ax.legend()
	axs.set_xlabel("Resolution (\$px\$)")
    axs.set_ylabel("Time (\$s\$)")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(plotsdir(
        string("bench_dim", batch_),
        string(file, "_para_", test_name[test_no], "_time.png"),
    ))
    close("all")

	# memory
	fig, ax = plt.subplots()
    for bm ∈ enumerate(benchm_gpu)
        ax.scatter(
            bm[2].memory * 1e-6,
            test_name_plt[bm[1]],
			olor=Utils.colours[1],
            label = "GPU",
            alpha = 0.3,
            edgecolors = "none",
        )
    end

    ax.legend()
	axs.set_xlabel("Resolution (\$px\$)")
    axs.set_ylabel("Memory")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(plotsdir(
        string("bench_dim", batch_),
        string(file, "_para_", test_name[test_no], "_time.png"),
    ))
    close("all")
end
end
