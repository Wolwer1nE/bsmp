module RHSGenerator

using LinearAlgebra
using MatrixMarket
using Printf

export generate_rhs, save_rhs, process_directory

"""
    generate_rhs(n)

Generate a Vector of `n` floats with 6 digits precision.
"""
function generate_rhs(n::Int)
    return round.(rand(n), digits=6)
end

"""
    save_rhs(rhs, dirpath, filename)

Save `rhs` Vector to `dirpath` directory with `filename` filename. 
"""
function save_rhs(rhs::Vector{Float64}, dirpath::String, filename::String)
    if !isdir(dirpath)
        mkpath(dirpath)
    end
    path = joinpath(dirpath, filename)
    open(path, "w") do io
        for val in rhs
            # print exactly 6 digits
            @printf(io, "%.6f\n", val)
        end
    end

    return path
end 

"""
    process_dir(dirpath)

Read all matrices in `dirpath` directory, generate right-hand side for them and save alongside
"""
function process_dir(dirpath::String)
    mtx_files = filter(f -> endswith(f, ".mtx"), readdir(dirpath))
    for file in mtx_files
        mtx_name = splitext(file)[1]
        mtx_path = joinpath(dirpath, file)
        # calling mmread with infoonly flag
        rows, cols, nnz = MatrixMarket.mmread(mtx_path, true)
        rhs = generate_rhs(cols)
        save_rhs(rhs, dirpath, mtx_name * ".rhs")
    end
end

end # RHSGenerator