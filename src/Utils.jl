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
using Images

# saving plots
location="../out/"


function save_orientations(A::Array, name::String, bright=1, loc=location, filetype=".png")
    for k in 1:size(A)[3]
        fn = string(loc,name,"_",k,filetype)
        save(fn, Gray.(bright .*(A[:,:,k])))
    end
end


function save_2d(A::Array, name::String, bright=1, loc=location, filetype=".png")
        fn = string(loc,name,filetype)
        save(fn, Gray.(bright .*(A)))
end

function save_2d_list(A::Array, name::String, bright=1, loc=location, filetype=".png")
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