# FIXME: AI SCRIPT! TO BE INSPECTED AND UPDATED!
# TODO: add multiple metrics to plot automatically
module PerformancePlotter

using CSV
using DataFrames
using Makie

export plot_csv

function plot_csv(filename::String; col="avg_time")
    # Read the CSV file
    df = CSV.read(filename, DataFrame, delim=';')
    
    # Convert column names to strings for consistent handling
    rename!(df, [string(name) for name in names(df)])
    
    # Convert matrix_name to String if it's not already
    if eltype(df.matrix_name) <: AbstractString
        df.matrix_name = String.(df.matrix_name)
    end
    
    # Ensure the column exists
    if !(col in names(df))
        error("Column '$col' not found. Available columns: $(names(df))")
    end
    
    # Get the data to plot
    x_data = 1:nrow(df)  # Use numeric positions for x-axis
    x_labels = df.matrix_name
    y_data = df[!, col]
    
    # Convert y_data to Float64 if it's not numeric
    if !(eltype(y_data) <: Number)
        y_data = parse.(Float64, y_data)
    end
    
    # Create the plot
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
              title="Bar Plot",
              xlabel="Matrix Name",
              ylabel="Average Time (s)",
            )
    
    # Use barplot with numeric x positions
    barplot!(ax, x_data, y_data, color=:blue)
    
    # Set custom x-axis ticks and labels
    ax.xticks = (x_data, x_labels)
    ax.xticklabelrotation = π/4
    
    return fig
end

end # module