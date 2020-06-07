#=
kernel_plots:
- Julia version: 1.4.0
- Author: niallcullinane
- Date: 2020-06-07
=#
using Plots
using Utils

testa(2)

use_intellij_backend()
1+1
x = 1:10; y = rand(10); # These are the plotting data
plot(x, y)




Template(;user="nialltc",
    plugins=[
        License(; name="MPL"),
        Git(; manifest=true, ssh=true),
        GitHubActions(; x86=true),
        TravisCI(),
        Codecov(),
        Documenter{GitHubActions}(),
        Develop(), Citation()], manifest = true)
