using Dolo
# Bernouilli law

struct Bernouilli <: Dolo.DiscreteProcess
    Π::Float64
end

Bernouilli(;Π=0.5) = Bernouilli(Π)

function discretize(bernouilli::Bernouilli)
    x = reshape([0., 1.], 2, 1)
    w = [1 - bernouilli.Π, bernouilli.Π]
    return Dolo.DiscretizedIIDProcess(x, w)
end

discretize(Bernouilli(0.6))

BernouilliSampler(bernouilli::Bernouilli, n::Int64) = Float64.(rand(n).> 1. - bernouilli.Π)
BernouilliSampler(Bernouilli(),5)


mutable struct Mixture <: Dolo.ContinuousProcess
    index::Dolo.DiscreteProcess
    distributions :: NTuple{2, <: Dolo.ContinuousProcess}
end


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

discretize(Mixture(Bernouilli(0.1), tuple(Dolo.MvNormal(0.4))))

Mixture7(Bernouilli(0.1), tuple(Dolo.MvNormal(0.4), Dolo.MvNormal(0.8)))

mutable struct Mixture3 <: Dolo.ContinuousProcess
    index::Dolo.DiscreteProcess
    distributions:: Tuple{<:Dolo.ContinuousProcess}
end

mutable struct Mixture6 <: Dolo.ContinuousProcess
    index::Dolo.DiscreteProcess
    distributions:: Tuple{Float64, Float64}
end

Mixture6(Bernouilli(),tuple(0.5,0.3))

Tuple(0.5,0.3)