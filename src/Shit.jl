include("Initializers.jl")
include("Math.jl")
include("Utilities.jl")

using Random, Statistics, LinearAlgebra

using Main.Initializers
using Main.Math
using Main.Utilities

macro sayhello1(name::String)
   return :( println("Hello, ", $name, "!") )
end

macro sayhello2(name::Symbol)
   return :( println("Hello, ", $name, "!") )
end

macro sayhello3(name::Int)
   return :( println("Hello, ", $name, "!") )
end

aaa = 12
@sayhello1 "aaa"
@sayhello2 aaa
@sayhello3 64
# macro make_struct(name)

@macroexpand @implement_layer_cache_struct("PPPP", Type{Int32}, Type{Float32})
@implement_layer_cache_struct("Mato", Type{Int32}, Type{Float32})


a = randn(4,4)
b = a

b[b .< 0] .= 0
b
a
a
diagm(0 => a)


function main()
   function lol(type::Type, dims::Dims{N} where N)
      dims
   end

   lol(Float32, 1,2)
end

main()




diagm(randn(3))


i = 7
k = 3
s = 3

j = 2 * (k - 1) + i

o = (j - k) / s + 1
floor(Int64, o)

compute_convolution_output_size_2d((7, 7), 3, 4, "valid")
