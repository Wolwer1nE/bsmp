#!/usr/bin/env ruby
# Simple block-diagonal matrix generator.
# Usage:
#   ruby generators/matrix.rb <output_file> [n_blocks=1000] [block_size=4] [--random]
# Produces a block-diagonal sparse matrix in triplet form:
#   row,col,value per line (0-based indices)
# Also produces an RHS vector file: rhs_<output_file>

if ARGV.length < 1 || ARGV.include?("-h") || ARGV.include?("--help")
  warn "Usage: ruby generators/matrix.rb <output_file> [n_blocks=1000] [block_size=4] [--random]"
  exit 1
end

outfile    = ARGV[0]
n_blocks   = (ARGV[1] || "1000").to_i
block_size = (ARGV[2] || "4").to_i
use_random = ARGV.include?("--random")

unless n_blocks > 0 && block_size > 0
  warn "n_blocks and block_size must be positive integers"
  exit 1
end

unless File.directory?(outfile)
  warn "outfile directory must exist"
  exit 1
end

File.open(outfile, "wb") do |f|
  n_blocks.times do |i|
    block_start = i * block_size
    value = use_random ? (rand * 1.0) : (i + 1)
    block_size.times do |j|
      block_size.times do |k|
        row = block_start + j
        col = block_start + k
        f.write("#{row} #{col} #{value}\n")
      end
    end
  end
end

n_rows = block_size * n_blocks
rhs_outfile = File.join(File.dirname(outfile), "rhs_#{File.basename(outfile)}")
File.open(rhs_outfile, "wb") do |f|
  n_rows.times do |i|
    f.write("#{1 + i / block_size}\n")
  end
end

STDERR.puts "Generated matrix '#{outfile}' with #{n_blocks} blocks of size #{block_size}."
STDERR.puts "RHS written to rhs_#{outfile}."
