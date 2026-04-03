module PerformancePlotter

using CSV, DataFrames, CategoricalArrays, Makie

export plot_metric, read_all_data

"""
    read_data(file)

Reads performance data from `file` path given.

An :alg_name column is added to data before return. It is the name of the file.

# Arguments
- `file`: path to a .csv file 

# Returns
- `data`: DataFrame from the file with additional :alg_name column 
"""
function read_data(file)
    file_df = CSV.read(file, DataFrame)
    insertcols!(file_df, :alg_name => splitext(basename(file))[1])
    return file_df
end

"""
    read_all_data(dir)

Reads performance data from all files in `dir` and merges them.

All data files must have the same set of columns.

It uses read_data, so :alg_name is added for each file.

# Arguments
- `dir`: path to a dir with .csv files.

# Returns
- `data`: DataFrame with all the files 
"""
function read_all_data(dir)
    files = filter(f -> occursin(r".*\.csv$", f), readdir(dir))
    df = DataFrame()

    for file in files
        full_path = joinpath(dir, file)
        append!(df, read_data(full_path))
    end

    return df
end

"""
    plot_metric(data, metric=:avg_time)

Plots metric `metric` from the DataFrame `data`.

Creates multiple small plots, one for each matrix name.

# Arguments
- `data`: DataFrame with performance data Required columns in data:
    - :matrix_name
    - :alg_name
    - `metric`
- `metric`: Symbol of a metric needed to be plotted. By default it is `:avg_time`

# Returns
- `fig`: Makie figure with multiple barplots, one per matrix name  
"""
function plot_metric(data, metric=:avg_time)

    for col in [:matrix_name, :alg_name, metric]
        hasproperty(data, col) || error("No column $col was found in the data")
    end

    if !(eltype(data[!, metric]) <: Number)
        data[!, metric] = parse.(Float64, data[!, metric])
    end

    # Get unique
    matrices = unique(data.matrix_name)
    n_matrices = length(matrices)
    algorithms = unique(data.alg_name)
    n_algorithms = length(algorithms)
    
    
    # Calculate grid dimensions
    n_cols = min(3, n_matrices)
    n_rows = ceil(Int, n_matrices / n_cols)
    
    colors = cgrad(:alpine, n_algorithms, categorical = true)
    
    # Create figure with more vertical space for rotated labels
    fig = Figure(size = (1200, 400 * n_rows + 67))
    
    # Create a subplot for each matrix
    for (i, matrix) in enumerate(matrices)
        # Filter data for current matrix
        matrix_data = filter(row -> row.matrix_name == matrix, data)
        
        # Sort the matrix data by the metric value (ascending order)
        sorted_data = sort(matrix_data, [metric], rev = false)
        
        sorted_algorithms = sorted_data.alg_name
        
        x_positions = 1:length(sorted_algorithms)
        
        # Create subplot
        row = div(i - 1, n_cols) + 1
        col = (i - 1) % n_cols + 1
        
        ax = Axis(fig[row, col],
            title = matrix,
            ylabel = i > (n_rows - 1) * n_cols ? string(metric) : "",
            xlabel = i > (n_rows - 1) * n_cols ? "Algorithm" : "",
            xminorticksvisible = true,
            xminorgridvisible = true,
            xticks = (x_positions, string.(sorted_algorithms)),
            xticklabelrotation = π/4,  # Rotate x-labels by 45 degrees
            xticklabelsize = 10,
            yticklabelsize = 10,
            titlesize = 14,
        )
        
        bp = barplot!(
            ax,
            x_positions,
            sorted_data[!, metric],
            color = [colors[findfirst(==(alg), algorithms)] for alg in sorted_algorithms],
            direction = :y,
            flip_labels_at = 0.85,
            bar_labels = :y,
            label_size = 10,
            color_over_background = :black,
            color_over_bar = :white,
            strokewidth = 1,
            strokecolor = :black,
        )
        
        ax.xminorgridvisible = false
        ax.yminorgridvisible = true
        ax.yminorgridcolor = (:gray, 0.3)
        ax.yminorgridwidth = 0.5
    end
    
    legend_elements = [
        PolyElement(color = colors[i], strokecolor = :black, strokewidth = 1) for i in 1:n_algorithms
    ]
    
    if n_matrices <= 3
        Legend(fig[1, n_cols+1], legend_elements, string.(algorithms), "Algorithm",
               framevisible = true, framecolor = (:gray, 0.5))
    else
        Legend(fig[n_rows+1, 1:n_cols], legend_elements, string.(algorithms), "Algorithm",
               orientation = :horizontal, framevisible = true, framecolor = (:gray, 0.5),
               tellheight = false, margin = (10, 10, 10, 10))
    end
    
    fig[0, :] = Label(fig, "Performance Comparison Across Matrices", fontsize = 18, font = :bold)
    
    return fig
end

function plot_metric_singleplot(data, metric=:avg_time)

    for col in [:matrix_name, :alg_name, metric]
        hasproperty(data, col) || error("No column $col was found in the data")
    end

    if !(eltype(data[!, metric]) <: Number)
        data[!, metric] = parse.(Float64, data[!, metric])
    end

    # CategoricalArray
    alg_cat = categorical(data.alg_name)
    mat_cat = categorical(data.matrix_name)

    # Get Integers from CategoricalArray
    data.alg_code = levelcode.(alg_cat)
    data.matrix_code = levelcode.(mat_cat)

    # Get Strings from CategoricalArray (same order as Integer levelcodes)
    algorithms = levels(alg_cat)
    matrices = levels(mat_cat)

    x = data.matrix_code
    colors = cgrad(:darkrainbow, length(algorithms), categorical = true)

    fig = Figure()
    ax = Axis(fig[1, 1],
        ylabel = "Матрица",
        xlabel = string(metric),
        xminorticksvisible = true, 
        xminorgridvisible = true
    )

    barplot!(
        ax,
        x,
        data[!, metric],
        dodge = data.alg_code,
        color = colors[data.alg_code],
        direction = :x,
        flip_labels_at=0.85,
        bar_labels = :y,
        label_size = 10,
        color_over_background=:black,
        color_over_bar=:red,
    )

    ax.yticks = (1:length(matrices), string.(matrices))

    elements = [
        PolyElement(color = colors[i]) for i in 1:length(algorithms)
    ]

    Legend(fig[1, 2], elements, string.(algorithms), "Algorithm")

    return fig
end


end # PerformancePlotter