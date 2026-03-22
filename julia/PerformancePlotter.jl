module PerformancePlotter

using CSV, DataFrames, CategoricalArrays, Makie

export plot_metric, read_all_data

function read_data(file)
    file_df = CSV.read(file, DataFrame)
    insertcols!(file_df, :alg_name => splitext(basename(file))[1])
    return file_df
end

function read_all_data(dir)
    files = filter(f -> occursin(r".*\.csv$", f), readdir(dir))
    df = DataFrame()

    for file in files
        full_path = joinpath(dir, file)
        append!(df, read_data(full_path))
    end

    return df
end

function plot_metric(data, metric=:avg_time)

    # Validate columns
    for col in [:matrix_name, :alg_name, metric]
        hasproperty(data, col) || error("No column $col was found in the data")
    end

    # Ensure numeric metric
    if !(eltype(data[!, metric]) <: Number)
        data[!, metric] = parse.(Float64, data[!, metric])
    end

    # Categorical encoding (stable!)
    alg_cat = categorical(data.alg_name)
    mat_cat = categorical(data.matrix_name)

    data.alg_code = levelcode.(alg_cat)
    data.matrix_code = levelcode.(mat_cat)

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
        color_over_bar=:white,
    )

    ax.yticks = (1:length(matrices), string.(matrices))

    elements = [
        PolyElement(color = colors[i]) for i in 1:length(algorithms)
    ]

    Legend(fig[1, 2], elements, string.(algorithms), "Algorithm")

    return fig
end

end # PerformancePlotter