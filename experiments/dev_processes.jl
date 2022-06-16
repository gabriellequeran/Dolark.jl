# for processes.jl in Dolo

using Dolo

## Bernouilli law

struct Bernouilli <: Dolo.DiscreteProcess
    π::Float64
end

Bernouilli(;π=0.5) = Bernouilli(π)

import Dolo.discretize

function Dolo.discretize(bernouilli::Bernouilli)
    x = reshape([0., 1.], 2, 1)
    w = [1 - bernouilli.π, bernouilli.π]
    return Dolo.DiscretizedIIDProcess(x, w)
end

Dolo.discretize(Bernouilli(0.6))

BernouilliSampler(bernouilli::Bernouilli, n::Int64) = Float64.(rand(n).> 1. - bernouilli.π)
BernouilliSampler(Bernouilli(),5)

## Mixtures
mutable struct Mixture <: Dolo.ContinuousProcess
    index::Dolo.DiscreteProcess
    distributions :: Tuple{<: Dolo.ContinuousProcess, <: Dolo.ContinuousProcess}
end

Mixture(;index=Bernouilli(), distributions = Dict(Symbol("1") => Dolo.ConstantProcess(), Symbol("0") => Dolo.UNormal())) = Mixture(index, tuple(distributions[Symbol("0")], distributions[Symbol("1")]))

Mixture()

function Dolo.discretize(self::Mixture)
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

Dolo.discretize(Mixture(Bernouilli(0.1), tuple(Dolo.MvNormal(0.4), Dolo.MvNormal(0.8))))



## ConstantProcess

function Dolo.discretize(constantprocess::Dolo.ConstantProcess)
    μ = constantprocess.μ
    return Dolo.DiscretizedIIDProcess(reshape(μ,length(μ),1),[1. for i=1:length(μ)])
end

discretize(Dolo.ConstantProcess([0.4,0.3]))

## UNormal extended to enable the option of μ ≠ 0
import Dolo.UNormal
Dolo.UNormal(;μ=0.0, σ=0.0) = Dolo.MvNormal([μ], reshape([σ^2], 1, 1))

Dolo.UNormal(;μ=0.2, σ=0.)





## ProductProcess

mutable struct ProductProcess_{P1<:Dolo.AbstractProcess,P2<:Dolo.AbstractProcess, P3<:Dolo.AbstractProcess} <: Dolo.AbstractProcess
    process_1::P1
    process_2::P2
    process_3::P3
end

ProductProcess_(Dolo.ConstantProcess(), Dolo.UNormal(), Mixture())

ProductProcess_(p) = p

function Dolo.discretize(pp::ProductProcess_{Dolo.ConstantProcess, <:Dolo.IIDExogenous, Mixture}; opt=Dict())
        print(pp.process_1)
        print(pp.process_3)
        diidp2 = Dolo.discretize(pp.process_2)
        inodes2 = diidp2.integration_nodes
        diidp3 = Dolo.discretize(pp.process_3)
        inodes3 = diidp3.integration_nodes
        iit = hcat(
            vcat([pp.process_1.μ' for i=1:size(inodes2,1)*size(inodes3,1)]...), 
            vcat([inodes2[i,1] for j=1:size(inodes3,1) for i=1:size(inodes2,1)]...),
            vcat([inodes3 for i=1:size(inodes2,1)]...)
        )
        weights = vcat([diidp2.integration_weights[i]*diidp3.integration_weights[j] for j=1:size(inodes3,1) for i=1:size(inodes2,1)]...)
        return Dolo.DiscretizedIIDProcess(diidp2.grid, iit, weights)
end

Dolo.ConstantProcess([0.2])
Dolo.UNormal(;μ=0.5, σ=0.0004)
Mixture(Bernouilli(),tuple(Dolo.UNormal(;μ=0.5, σ=0.0004), Dolo.UNormal(;μ=0.6, σ=0.0003)))
pp0 = ProductProcess_(Dolo.ConstantProcess([0.2]), Dolo.UNormal(;μ=0.5, σ=0.0004), Mixture(Bernouilli(),tuple(Dolo.UNormal(;μ=0.5, σ=0.0004), Dolo.UNormal(;μ=0.6, σ=0.0003))))
Dolo.discretize(pp0)

# For domains.jl

import Dolo.get_domain

Dolo.get_domain(mixture::Mixture) = Dolo.get_domain(mixture.distributions[1])


function Dolo.get_domain(pp::ProductProcess_)
    dom1 = Dolo.get_domain(pp.process_1)
    dom2 = Dolo.get_domain(pp.process_2)
    dom3 = Dolo.get_domain(pp.process_3)
    return Dolo.ProductDomain(Dolo.ProductDomain(dom1, dom2),dom3)
end



# For Dolo.jl

minilang = Dolo.minilang
Dolo.add_language_elements!(minilang, Dict(
    "!Mixture"=>Mixture,
    "!Bernouilli"=>Bernouilli,
))

# For model.jl

import Dolo.get_exogenous

function Dolo.get_exogenous(data, exosyms, fcalib)

    calibration = fcalib

    if !("exogenous" in keys(data))
        return nothing
    end

    exo_dict = data[:exogenous]
    cond = !(exo_dict.tag=="tag:yaml.org,2002:map")

    syms = cat( [[Symbol(strip(e)) for e in split(k, ",")] for k in keys(exo_dict)]..., dims=1)
    
    expected = exosyms
    if (syms != expected)
        msg = string("In 'exogenous' section, shocks must be declared in the same order as shock symbol. Found: ", syms, ". Expected: ", expected, ".")
        throw(ErrorException(msg))
    end
    processes = []
    for k in keys(exo_dict)
        v = exo_dict[k]
        p = Dolang.eval_node(v, calibration, minilang, Dolang.ToGreek())
        push!(processes, p)
    end
    if length(processes) == 2
        return Dolo.ProductProcess(processes...)
    else
        return ProductProcess_(processes...)
    end
end