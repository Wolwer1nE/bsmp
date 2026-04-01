include("PerformancePlotter.jl")
using .PerformancePlotter
using CairoMakie

set_theme!(theme_latexfonts())

matsets = ["cylshell"]

for matset in matsets
    data = read_all_data("../output/$matset/")

    println(names(data))

    fig = plot_metric(data)

    # Create output folder and save
    mkpath("../output/$matset/plots")
    save(joinpath("plots", "barplot.png"), fig)
    save(joinpath("plots", "barplot.pdf"), fig)
    save(joinpath("plots", "barplot.svg"), fig)

    println("Done! Check the 'plots' folder.")
end