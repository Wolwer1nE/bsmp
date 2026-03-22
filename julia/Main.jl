# FIXME: AI SCRIPT! TO BE INSPECTED AND UPDATED!
#import Pkg
#Pkg.add(["CSV", "DataFrames", "Makie"])
# TODO: add dependency management as an isolated environment 

include("PerformancePlotter.jl")
using .PerformancePlotter
using CairoMakie

set_theme!(theme_latexfonts())

# Generate and save the plot
data = read_all_data("../output/cylshell/")

println(names(data))

fig = plot_metric(data)

# Create output folder and save
mkpath("plots")
save(joinpath("plots", "barplot.png"), fig)
save(joinpath("plots", "barplot.pdf"), fig)
save(joinpath("plots", "barplot.svg"), fig)

println("Done! Check the 'plots' folder.")