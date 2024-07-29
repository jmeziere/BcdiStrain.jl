module BcdiStrain
    using CUDA
    using LinearAlgebra
    using Statistics
    using BcdiCore
    using BcdiTrad

    include("State.jl")
    include("Operators.jl")
end
