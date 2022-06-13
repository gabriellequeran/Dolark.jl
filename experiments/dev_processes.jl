# for processes.jl in Dolo

using Dolo

## Bernouilli law

struct Bernouilli <: Dolo.DiscreteProcess
    π::Float64
end

Bernouilli(;π=0.5) = Bernouilli(π)

function discretize(bernouilli::Bernouilli)
    x = reshape([0., 1.], 2, 1)
    w = [1 - bernouilli.π, bernouilli.π]
    return Dolo.DiscretizedIIDProcess(x, w)
end

discretize(Bernouilli(0.6))

BernouilliSampler(bernouilli::Bernouilli, n::Int64) = Float64.(rand(n).> 1. - bernouilli.π)
BernouilliSampler(Bernouilli(),5)

## Mixtures
mutable struct Mixture <: Dolo.ContinuousProcess
    index::Dolo.DiscreteProcess
    distributions :: Tuple{<: Dolo.ContinuousProcess, <: Dolo.ContinuousProcess}
end

Mixture(;index=Bernouilli(), distributions = Dict(Symbol("1") => Dolo.ConstantProcess(), Symbol("0") => Dolo.UNormal())) = Mixture(index, tuple(distributions[Symbol("0")], distributions[Symbol("1")]))

Mixture()

function discretize(self::Mixture)
    inddist = discretize(self.index)
    nodes = []
    weights = []
    for i in 1:(Dolo.n_inodes(inddist, 0))
        wind =  Dolo.iweight(inddist, 0, i)
        # xind = inddist.inode(0, i)
        dist = Dolo.discretize(self.distributions[i])
        for j in 1:(Dolo.n_inodes(dist, 0))
                w = Dolo.iweight(dist, 0, j)
                x = Dolo.inode(dist, 0, j)
                append!(nodes, x)
                append!(weights, wind * w)
        end
    end
    nodes = reshape(nodes, length(nodes), 1)
    return Dolo.DiscretizedIIDProcess(nodes, weights)
end

discretize(Mixture(Bernouilli(0.1), tuple(Dolo.MvNormal(0.4), Dolo.MvNormal(0.8))))



## UNormal extended to enable the option of μ ≠ 0
import Dolo.UNormal
Dolo.UNormal(;μ=0.0, σ=0.0) = Dolo.MvNormal([μ], reshape([σ^2], 1, 1))

Dolo.UNormal(;μ=0.2, σ=0.)







# For Dolo.jl

minilang = Dolo.minilang
Dolo.add_language_elements!(minilang, Dict(
    "!Mixture"=>Mixture,
    "!Bernouilli"=>Bernouilli,
))