"""
# module utils

- Julia version: 1.4
- Author: niallcullinane
- Date: 2020-06-07

# Examples

```jldoctest
julia>
```
"""
module Utils
using PyPlot, Images
using DrWatson
@quickactivate "MEngProject"

# saving plots
location=plotsdir()

layers = ["L6 (\$x\$)","L6(\$x\$)","L4 excit (\$y\$)","L4 excit (\$y\$)","L4 inhib (\$m\$)","L4 inhib (\$m\$)","L2/3 excit (\$z\$)","L2/3 excit (\$z\$)","L2/3 inhib (\$s\$)","L2/3 inhib (\$s\$)","LGN (\$v^+\$)","LGN (\$v^-\$)"]
la = ["x","x","y","y","m","m","z","z","s","s","vp","vm"]

function plot_rb(img::AbstractArray;  name="img", save = false, axMin = -1, axMax = 1, clbar=false,  loc=location, filetype=".png")
    findmax(img)[1] > axMax && throw(ArgumentError(string("Image has max ", findmax(img)[1], ",outside range")))
    findmin(img)[1] < axMin && throw(ArgumentError(string("Image has min ", findmin(img)[1], ",outside range")))
    fig, ax = plt.subplots()

    im = ax.imshow(img, cmap=matplotlib.cm.RdBu_r,
               vmax=axMax, vmin=axMin)
    if clbar
        cbar = fig.colorbar(im,  shrink=0.9, ax=ax)
    end

    plt.axis("off")
    fig.tight_layout()
    plt.show()
    if save
        plt.savefig(string(loc,name,filetype))
    end
end

function plot_two(img::AbstractArray, k, t;   name="img", save = false, axMin = -1, axMax = 1, clbar=false,  loc=location, filetype=".png")
    findmax(img)[1] > axMax && throw(ArgumentError(string("Image has max ", findmax(img)[1], ",outside range")))
    findmin(img)[1] < axMin && throw(ArgumentError(string("Image has min ", findmin(img)[1], ",outside range")))
    fig, ax = plt.subplots()

    im = ax.imshow(img[:,:,k,1,t], cmap=matplotlib.cm.BrBG,
               vmax=axMax, vmin=axMin)
	ax.imshow(img[:,:,k+1,1,t], cmap=matplotlib.cm.BrBG,
               vmax=axMax, vmin=axMin)
    if clbar
        cbar = fig.colorbar(im,  shrink=0.9, ax=ax)
    end

    plt.axis("off")
    fig.tight_layout()
    plt.show()
    if save
        plt.savefig(string(loc,name,filetype))
    end
end

function plot_gs(img::AbstractArray;  name="img", save = false, axMin = 0, axMax = 2, clbar=false,  loc=location, filetype=".png")
    findmax(img)[1] > axMax && throw(ArgumentError(string("Image has max ", findmax(img)[1], ",outside range")))
    findmin(img)[1] < axMin && throw(ArgumentError(string("Image has min ", findmin(img)[1], ",outside range")))
    fig, ax = plt.subplots()

    im = ax.imshow(img, cmap=matplotlib.cm.gray,
               vmax=axMax, vmin=axMin)
    if clbar
        cbar = fig.colorbar(im,  shrink=0.9, ax=ax)
    end

    plt.axis("off")
    fig.tight_layout()
    plt.show()
    if save
        plt.savefig(string(loc,name,filetype))
    end
end


function save_orientations_rb(A::Array, name::String,  axMin=-1, axMax=1, clbar=false, loc=location, filetype=".png")
    for k in 1:size(A)[3]
        fn = string(name,"_",k)
        plot_rb(A[:,:,k], fn, true, axMin, axMax, clbar,  loc, filetype)
    end
end


function save_orientations_gs(A::Array, name::String,  axMin=0, axMax=1, clbar=false, loc=location, filetype=".png")
    for k in 1:size(A)[3]
        fn = string(name,"_",k)
        plot_gs(A[:,:,k], fn, true, axMin, axMax, clbar,  loc, filetype)
    end
end


function save_2d_list_rb(A::Array, name::String, loc=location, filetype=".png", axMin=-1, axMax=1, clbar=false)
n=0
    for img in A
        n+=1
        plot_101_rb(img,  name, true, axMin, axMax, clbar,  loc, filetype)
    end
end


function save_orientations_gs_(A::Array, name::String, bright=1, loc=location, filetype=".png")
    for k in 1:size(A)[3]
        fn = string(loc,name,"_",k,filetype)
        save(fn, Gray.(bright .*(A[:,:,k])))
    end
end


function save_2d_gs(A::Array, name::String, bright=1, loc=location, filetype=".png")
        fn = string(loc,name,filetype)
        save(fn, Gray.(bright .*(A)))
end

function save_2d_list_gs(A::Array, name::String, bright=1, loc=location, filetype=".png")
n=0
    for img in A
        n+=1
        save_2d(img, string(name,n), bright, loc, filetype)
    end
end

function testa(x)
    x*2
    end


end
