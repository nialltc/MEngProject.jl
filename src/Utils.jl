"""
# module utils

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-06-07

Functions for plotting.
# Examples

```jldoctest
julia>
```
"""
module Utils
using PyPlot, Images, Statistics
using DrWatson
@quickactivate "MEngProject"

# saving plots
location = plotsdir()

# pretty layer names
layers = [
    "L6 (\$x\$)",
    "L6(\$x\$)",
    "L4 excit (\$y\$)",
    "L4 excit (\$y\$)",
    "L4 inhib (\$m\$)",
    "L4 inhib (\$m\$)",
    "L2/3 excit (\$z\$)",
    "L2/3 excit (\$z\$)",
    "L2/3 inhib (\$s\$)",
    "L2/3 inhib (\$s\$)",
    "LGN (\$v\$)",
    "LGN (\$v\$)",
]

# bare layer names
layers_1 = [
    "\$x_1\$",
    "\$x_2\$",
    "\$y_1\$",
    "\$y_2\$",
    "\$m_1\$",
    "\$m_2\$",
    "\$z_1\$",
    "\$z_2\$",
    "\$s_1\$",
    "\$s_2\$",
    "\$v^+\$",
    "\$v^-\$",
]

# short layer names
la = ["x", "x", "y", "y", "m", "m", "z", "z", "s", "s", "v", "v"]

# line types for matplotlib
lines = [
    "b--",
    "g--",
    "r-.",
    "c-.",
    "r:",
    "c:",
    "m-",
    "y-",
    "m:",
    "y:",
    "b-.",
    "g-.",
]

# colours for matplotlib
colours = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


"""
Creates image plot with blue-red cmap.
"""
function plot_rb(
    img::AbstractArray;
    name = "img",
    save = false,
    axMin = -1,
    axMax = 1,
    clbar = false,
    loc = location,
    filetype = ".png",
)
    findmax(img)[1] > axMax && throw(ArgumentError(string(
        "Image has max ",
        findmax(img)[1],
        ",outside range",
    )))
    findmin(img)[1] < axMin && throw(ArgumentError(string(
        "Image has min ",
        findmin(img)[1],
        ",outside range",
    )))
    fig, ax = plt.subplots()

    im = ax.imshow(img, cmap = matplotlib.cm.RdBu_r, vmax = axMax, vmin = axMin)
    if clbar
        cbar = fig.colorbar(im, shrink = 0.9, ax = ax)
    end

    plt.axis("off")
    fig.tight_layout()
    plt.show()
    if save
        plt.savefig(string(loc, name, filetype))
    end
end


"""
Creates image plot with two images overlayed.
"""
function plot_two(
    img::AbstractArray,
    k,
    t;
    name = "img",
    save = false,
    axMin = -1,
    axMax = 1,
    clbar = false,
    loc = location,
    filetype = ".png",
)
    findmax(img)[1] > axMax && throw(ArgumentError(string(
        "Image has max ",
        findmax(img)[1],
        ",outside range",
    )))
    findmin(img)[1] < axMin && throw(ArgumentError(string(
        "Image has min ",
        findmin(img)[1],
        ",outside range",
    )))
    fig, ax = plt.subplots()

    im = ax.imshow(
        img[:, :, k, 1, t],
        cmap = matplotlib.cm.BrBG,
        vmax = axMax,
        vmin = axMin,
    )
    ax.imshow(
        img[:, :, k+1, 1, t],
        cmap = matplotlib.cm.BrBG,
        vmax = axMax,
        vmin = axMin,
    )
    if clbar
        cbar = fig.colorbar(im, shrink = 0.9, ax = ax)
    end

    plt.axis("off")
    fig.tight_layout()
    plt.show()
    if save
        plt.savefig(string(loc, name, filetype))
    end
end


"""
Creates image plot with grey cmap.
"""
function plot_gs(
    img::AbstractArray;
    name = "img",
    save = false,
    axMin = 0,
    axMax = 2,
    clbar = false,
    loc = location,
    filetype = ".png",
)
    findmax(img)[1] > axMax && throw(ArgumentError(string(
        "Image has max ",
        findmax(img)[1],
        ",outside range",
    )))
    findmin(img)[1] < axMin && throw(ArgumentError(string(
        "Image has min ",
        findmin(img)[1],
        ",outside range",
    )))
    fig, ax = plt.subplots()

    im = ax.imshow(img, cmap = matplotlib.cm.gray, vmax = axMax, vmin = axMin)
    if clbar
        cbar = fig.colorbar(im, shrink = 0.9, ax = ax)
    end

    plt.axis("off")
    fig.tight_layout()
    plt.show()
    if save
        plt.savefig(string(loc, name, filetype))
    end
end


"""
Plots all orientations with blue-red cmap.
"""
function save_orientations_rb(
    A::Array,
    name::String,
    axMin = -1,
    axMax = 1,
    clbar = false,
    loc = location,
    filetype = ".png",
)
    for k = 1:size(A)[3]
        fn = string(name, "_", k)
        plot_rb(A[:, :, k], fn, true, axMin, axMax, clbar, loc, filetype)
    end
end


"""
Plots all orientations with gray cmap.
"""
function save_orientations_gs(
    A::Array,
    name::String,
    axMin = 0,
    axMax = 1,
    clbar = false,
    loc = location,
    filetype = ".png",
)
    for k = 1:size(A)[3]
        fn = string(name, "_", k)
        plot_gs(A[:, :, k], fn, true, axMin, axMax, clbar, loc, filetype)
    end
end



# function save_2d_list_rb(
#     A::Array,
#     name::String,
#     loc = location,
#     filetype = ".png",
#     axMin = -1,
#     axMax = 1,
#     clbar = false,
# )
#     n = 0
#     for img in A
#         n += 1
#         plot_101_rb(img, name, true, axMin, axMax, clbar, loc, filetype)
#     end
# end
#
#
# function save_orientations_gs_(
#     A::Array,
#     name::String,
#     bright = 1,
#     loc = location,
#     filetype = ".png",
# )
#     for k = 1:size(A)[3]
#         fn = string(loc, name, "_", k, filetype)
#         save(fn, Gray.(bright .* (A[:, :, k])))
#     end
# end
#
#
# function save_2d_gs(
#     A::Array,
#     name::String,
#     bright = 1,
#     loc = location,
#     filetype = ".png",
# )
#     fn = string(loc, name, filetype)
#     save(fn, Gray.(bright .* (A)))
# end
#
# function save_2d_list_gs(
#     A::Array,
#     name::String,
#     bright = 1,
#     loc = location,
#     filetype = ".png",
# )
#     n = 0
#     for img in A
#         n += 1
#         save_2d(img, string(name, n), bright, loc, filetype)
#     end
# end


"""
Plots activation vs time with all layers and orientations.
Uses highest value pixel at end k=1 orientation for each layer.
"""
function plot_t_act(sol, name, batch, file; save = true)
    fig, axs = plt.subplots()
    @inbounds begin
        for k ∈ 1:12
            v1 = @view sol[:, :, k, 1, end]
            v2 = @view sol[findmax(v1)[2][1], findmax(v1)[2][2], k, 1, :]
            layer = Utils.layers_1[k]
            axs.plot(sol.t, v2, Utils.lines[k], label = "$layer", alpha = 0.8)
        end
        axs.set_xlabel("Time")
        axs.set_ylabel("Activation")
        plt.legend()
        fig.tight_layout()
        if save
            plt.savefig(plotsdir(
                string(name, batch),
                string(file, "_time.png"),
            ))
        end
        close("all")
    end
    return nothing
end


"""
Plots mean activation vs time with all layers and orientations.
Uses mean for each layer/orientation.
Currently very slow.
"""
function plot_t_act_mean(sol, name, batch, file; save = true)
    fig, axs = plt.subplots()
    @inbounds begin
        for k ∈ 1:12
            # v1 = @view sol[:, :, k, 1, end]
            # v2 = @view sol[findmax(v1)[2][1], findmax(v1)[2][2], k, 1, :]
			v2 = Array{eltype(float32)}(undef, size(sol, 5))
			for s in 1:size(sol, 5)
				v2[s] = mean(@view sol[:,:, k, 1, s])
			end
			layer = Utils.layers_1[k]
			axs.plot(sol.t, v2, Utils.lines[k], label = "$layer", alpha = 0.8)
        end
        axs.set_xlabel("Time")
        axs.set_ylabel("Activation")
        plt.legend()
        fig.tight_layout()
        if save
            plt.savefig(plotsdir(
                string(name, batch),
                string(file, "_time.png"),
            ))
        end
        close("all")
    end
    return nothing
end


"""
Plots activation vs time with all layers and orientations.
Uses highest value pixel at end of specified layer/orient.
Default is z, k=1
"""
function plot_t_act_px(sol, name, batch, file; d=7, save = true)
    fig, axs = plt.subplots()
    @inbounds begin
        v1 = @view sol[:, :, d, 1, end]
        for k ∈ 1:12
            v2 = @view sol[findmax(v1)[2][1], findmax(v1)[2][2], k, 1, :]
            layer = Utils.layers_1[k]
            axs.plot(sol.t, v2, Utils.lines[k], label = "$layer", alpha = 0.8)
        end
        axs.set_xlabel("Time")
        axs.set_ylabel("Activation")
        plt.legend()
        fig.tight_layout()
        if save
            plt.savefig(plotsdir(
                string(name, batch),
                string(file, "_time.png"),
            ))
        end
        close("all")
    end
    return nothing
end



"""
Plots activation vs time with all layers and orientations.
Takes in pixel address.
Default: (50,50)
Default is z, k=1
"""
function plot_t_act_spec(sol, name, batch, file; px=(50,50), save = true)
    fig, axs = plt.subplots()
    @inbounds begin
#         v1 = @view sol[:, :, d, 1, end]
        for k ∈ 1:12
            v2 = @view sol[px[1], px[2], k, 1, :]
            layer = Utils.layers_1[k]
            axs.plot(sol.t, v2, Utils.lines[k], label = "$layer", alpha = 0.8)
        end
        axs.set_xlabel("Time")
        axs.set_ylabel("Activation")
        plt.legend()
        fig.tight_layout()
        if save
            plt.savefig(plotsdir(
                string(name, batch),
                string(file, "_time.png"),
            ))
        end
        close("all")
    end
    return nothing
end


"""
Plots two orientantions together for all layers.
v^+ and v^- plotted together
Cbar alpha for lower image adjusted.
"""
function plot_k2(sol, t, name, batch, file; save = true, cb = true)

    @inbounds begin

        v0 = @view sol(t)[:, :, :, 1]
        axMax = findmax(v0)[1]

        # plot x, y, z, m, s
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
            if cb
                cbar = fig.colorbar(im2, shrink = 0.9, ax = ax)
                cbar.ax.set_xlabel("\$k=2\$")
                cbar = fig.colorbar(im, shrink = 0.9, ax = ax)
                cbar.set_alpha(0.5)
                cbar.draw_all()
                cbar.ax.set_xlabel("\$k=1\$")
            end
            layer = Utils.layers[k]
            plt.title("Layer: $layer, \$t=$t\$")
            plt.axis("off")
            fig.tight_layout()
            if save
                plt.savefig(plotsdir(
                    string(name, batch),
                    string(file, "_", t, "_", Utils.la[k], ".png"),
                ))
            end
            close("all")
        end

        # plot lgn
        k = 11
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
        cbar.ax.set_xlabel("\$v^-\$")
        cbar = fig.colorbar(im, shrink = 0.9, ax = ax)
        cbar.ax.set_xlabel("\$v^+\$")
        cbar.set_alpha(0.5)
        cbar.draw_all()

        layer = Utils.layers[k]
        plt.title("Layer: $layer, \$t=$t\$")
        plt.axis("off")
        fig.tight_layout()
        if save
            plt.savefig(plotsdir(
                string(name, batch),
                string(file, "_", t, "_", Utils.la[k], ".png"),
            ))
        end
        close("all")
    end
    return nothing
end

"""
Plots two orientantions together for all layers.
v^+ and v^- plotted sep in gs
Cbar alpha for lower image adjusted.
"""
function plot_k2_vsep(sol, t, name, batch, file; save = true, cb = true)

    @inbounds begin

        v0 = @view sol(t)[:, :, :, 1]
        axMax = findmax(v0)[1]

        # plot x, y, z, m, s
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
            if cb
                cbar = fig.colorbar(im2, shrink = 0.9, ax = ax)
                cbar.ax.set_xlabel("\$k=2\$")
                cbar = fig.colorbar(im, shrink = 0.9, ax = ax)
                cbar.set_alpha(0.5)
                cbar.draw_all()
                cbar.ax.set_xlabel("\$k=1\$")
            end
            layer = Utils.layers[k]
            plt.title("Layer: $layer, \$t=$t\$")
            plt.axis("off")
            fig.tight_layout()
            if save
                plt.savefig(plotsdir(
                    string(name, batch),
                    string(file, "_", t, "_", Utils.la[k], ".png"),
                ))
            end
            close("all")
        end

        # plot lgn
        k = 11
        fig, (ax, ax2) = plt.subplots(1,2)
        v1 = @view sol(t)[:, :, k, 1]
        v2 = @view sol(t)[:, :, k+1, 1]
        im = ax1.imshow(
            v1,
            cmap = matplotlib.cm.gray,
            vmax = axMax,
            vmin = 0,
        )
        im2 = ax2.imshow(
            v2,
            cmap = matplotlib.cm.gray,
            vmax = axMax,
            vmin = 0,
            # alpha = 0.5,
        )

        cbar = fig.colorbar(im2, shrink = 0.9, ax = ax)
        cbar.ax.set_xlabel("\$v^+, v^-\$")
        # cbar = fig.colorbar(im, shrink = 0.9, ax = ax)
        # cbar.ax2.set_xlabel("\$v^+\$")
        cbar.set_alpha(0.5)
        cbar.draw_all()

        layer = Utils.layers[k]
        plt.title("Layer: $layer, \$t=$t\$")
        plt.axis("off")
        fig.tight_layout()
        if save
            plt.savefig(plotsdir(
                string(name, batch),
                string(file, "_", t, "_", Utils.la[k], ".png"),
            ))
        end
        close("all")
    end
    return nothing
end
end
