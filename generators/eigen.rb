#!/usr/bin/env ruby
# General Eigen value problem generator
# Usage:
#   ruby generators/eigen.rb <output_file> [n] [density_A] [density_B] [--verbose]
# Produces two symmetric sparse matrices A and B in triplet form:
#   row,col,value per line (0-based indices)

require 'matrix'

n = argv[0] || 100
density_A = argv[0]  || 0.01 
density_B = argv[0]  || 0.005

def generate_symmetric_sparse_matrix(n, density, scale_factor = 1.0)

  triplets = []
  
  (0...n).each do |i|
    diag_value = (i + 1).to_f * scale_factor + rand * 5.0
    triplets << [i, i, diag_value]
  end
  
  num_nonzeros = (n * n * density / 2).to_i  
  
  num_nonzeros.times do
    i = rand(n)
    j = rand(n)
    next if i == j 
    next if i > j 
    
    value = (rand - 0.5) * 2.0 * scale_factor
    triplets << [i, j, value]
    triplets << [j, i, value]  
  end
  
  hash = {}
  triplets.each do |i, j, v|
    key = [i, j]
    hash[key] ||= 0.0
    hash[key] += v
  end

  result = []
  hash.each do |(i, j), v|
    result << [i, j, v] if v.abs > 1e-10
  end
  
  result.sort_by { |i, j, v| [i, j] }
end

def save_triplets(filename, triplets)
  File.open(filename, 'w') do |f|
    triplets.each do |i, j, v|
      f.puts "#{i} #{j} #{v}"
    end
  end
end

triplets_A = generate_symmetric_sparse_matrix(n, density_A, 1.0)
triplets_B = generate_symmetric_sparse_matrix(n, density_B, 0.5)

# B is reaaly good in this case, strongly diagonally dominant
triplets_B.each do |triplet|
  if triplet[0] == triplet[1]
    triplet[2] += 10.0
  end
end

# Сохраняем данные
puts
puts "Saving matrices..."
save_triplets("../data/matrix_A_1000.txt", triplets_A)
save_triplets("../data/matrix_B_1000.txt", triplets_B)

puts
puts "=" * 60
puts "Matrix statistics:"
puts "=" * 60
puts "Matrix A:"
puts "  Size: #{n} x #{n}"
puts "  Nonzeros: #{triplets_A.size}"
puts "  Density: #{(triplets_A.size.to_f / (n * n) * 100).round(3)}%"
puts "  Estimated memory: #{(triplets_A.size * 12 / 1024.0).round(2)} KB"
puts
puts "Matrix B:"
puts "  Size: #{n} x #{n}"
puts "  Nonzeros: #{triplets_B.size}"
puts "  Density: #{(triplets_B.size.to_f / (n * n) * 100).round(3)}%"
puts "  Estimated memory: #{(triplets_B.size * 12 / 1024.0).round(2)} KB"
puts