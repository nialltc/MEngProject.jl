using NNlib, ImageFiltering, Images
using DrWatson
using Statistics
using BenchmarkTools
@quickactivate "MEngProject"
imgLoc = datadir("img","stairs_100gs.png")


img = convert(Array{Float32,2}, load(imgLoc))

function reshape2d_4d(img::AbstractArray)
    reshape(img, size(img)[1], size(img)[2], 1, 1)
end


img = reshape2d_4d(img)

@benchmark mean(img[:,:,1,1])

Statistics.mean(img[:,:,1,1])

a = similar(img)
a = [1.,1.,1.,1.]

@benchmark mean!(a, img[:,:,1,1])


a




a = reshape(
    Array{eltype(img)}(undef, 100, 100 * 2),
    100,
    100,
    2,
    1,
)

a[:,:,1:1,:] .= img
a[:,:,2:2,:] .= img

b = Array{eltype(img)}(undef,  2)

for k in 1:2
b[k] = mean(a[:,:,k,:])
end

b
