using StaticArrays
import Dolark
import Dolo
using FiniteDiff
using LinearMaps
using LinearAlgebra
using Statistics
using Krylov


"""
Takes an evaluation of a model's state variables - y -  and the model itself to compute the difference between the offer and demand - A -
(and the decision rule) and, if specified by the argument diff, also returns, for a shock dy on y, the corresponding response dA of A. 
# Arguments
* `y::SVector{d,Float64}`: evaluation of the state variables
* `hmodel::HModel`: model with heterogeneous agents
# Optional Argument
* `init::boolean`:
* `dr0`: initial decision rule to start with in the time iteration algorithm
* `z::SVector{d',Float64}`: exogenous parameters
* `diff::boolean`: Indicates whether we want to compute differentials
* `dy::SVector{d,Float64}`: shock on y
* `smaxit::Int64`: maximum number of power iterations to inverse (I-A) as ΣA^i
* `tol_ν::Float64`: used to stop the iterations of power computations for inversions
# Returns
* `A::Vector{d,Float64}`: difference y_demand-y_offer
* `r`: decision rule
# Optionnaly, returns
* `dA::Vector{d,Float64}`: dA caused by dy
"""
function Ξ(y::SVector{d,Float64}, hmodel; it=0, dr0 = nothing, z=SVector(0.), diff=false, smaxit=1000, tol_ν=1e-10, log=nothing, log2 = nothing, t_∂A_∂y = 0.) where d
    
    parm = hmodel.calibration[:parameters]
    p, r_p_y = Dolark.projection(hmodel, Val{(0,1)}, y, z, parm)
    r, w = p
    p = SVector{length(p),Float64}(p...)

    t0 = time()
    Dolo.set_calibration!(hmodel.agent; r=r, w=w) # update the model to account for the correct values of p
    t_calib = time()-t0 # ~1e-2s, 1e-3s

    t0 = time()
    if it==0 # this disjunction of cases allows to speed up the time iteration algorithm when using the Newton method to find the 0 of Ξ[1]
        sol_agent = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
    else
        sol_agent = Dolo.improved_time_iteration(hmodel.agent; dr0 = dr0, verbose=false) # the second time Ξ is run, we start from dr0 as initial guess to speed up the computation
    end
    t_ITI = time()-t0 # ~1e-2s

    if !diff
        t0 = time()
        μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
        t_ergodic_dist = time()-t0 # ~6e-3s

        t0 = time()
        dmodel = Dolark.discretize(hmodel, sol_agent) 
        t_discretize = time()-t0 # ~1e-3s

        x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])
    
        # computation of A = K_demand - K_offer
        t0 = time()
        A = Dolark.𝒜(dmodel, μ, x, y, z; diff=false)
        t_A = time()-t0 # ~3e-4s
        return A, sol_agent.dr
    end

    t0 = time()
    μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
    t_ergodic_dist = time()-t0

    t0 = time()
    dmodel = Dolark.discretize(hmodel, sol_agent)
    t_discretize = time()-t0

    x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])

    t0 = time()
    μ, ∂G_∂μ, ∂G_∂x = dmodel.G(μ,x; diff=true) # here, μ is unchanged since it is already the ergodic distrib.
    t_G = time()-t0 # ~4e-2s

    # computation of A = K_demand - K_offer and of its derivatives
    t0 = time()
    A, ∂A_∂μ, ∂A_∂x, ∂A_∂y, ∂A_∂z = Dolark.𝒜(dmodel, μ, x, y, z; diff=true)
    t_A = time()-t0 # ~3e-4s


    t0 = time()
    function dA_(dy; log2 = log2, it = it)
        # computation of dp induced by dy
        t0 = time()
        dp = r_p_y * dy
        t_dp = time()-t0 # 3e-4s

        # computation of dx induced by dy
        t0 = time()
        J = Dolo.df_A(dmodel.F, x, x; exo=(p,p))
        L = Dolo.df_B(dmodel.F, x, x; exo=(p,p))
        F_p1, F_p2 = Dolo.df_e(dmodel.F, x, x, p, p)
        F_p = F_p1 + F_p2
        Dolo.mult!(L, -1.0) # L : -L
        Dolo.prediv!(L, J) # L : -J\L 
        π = - J \ F_p * dp
        dx = invert(L, π)
        t_dx = time()-t0 # ~8e-2s

        # computation of dμ induced by dy. ∂G/∂p MUST BE ADDED AND THE CONVERGENCE MUST BE CHECKED !!!
        t0 = time()
        dμ = invert(∂G_∂μ, ∂G_∂x * dx)
        t_dμ = time()-t0 # ~5e-3s

        # computation of dA induced by dy
        t0 = time()
        dA = convert(Matrix, ∂A_∂μ) * dμ + convert(Matrix,∂A_∂x) * dx + convert(Matrix,∂A_∂y) * dy 
        t_dA = time() - t0 # ~2e-3s

        if log2 != nothing
            Dolo.append!(log2; #verbose=verbose,
            it=it,
            t_dp= t_dp,
            t_dx= t_dx,
            t_dμ= t_dμ,
            t_dA= t_dA)
        end

        return(dA)
    end

    if log != nothing
        Dolo.append!(log; 
        it=it,
        calib= t_calib,
        ITI= t_ITI,
        ergodic_dist= t_ergodic_dist,
        discretize= t_discretize,
        G= t_G,
        A= t_A,
        ∂A_∂y= t_∂A_∂y)
    end

    return A, dA_, sol_agent.dr
end




function solve_agent_pb(hmodel; n_it=100, toll=1e-3, newton_adjustment_parameter = 0.2, krylov=false, log_tot = false, log_∂A_∂y = false) 

    y0, z0 = hmodel.calibration[:aggregate, :exogenous]
    y = SVector(y0...)
    z = SVector(z0...)
    N_y = length(y)

    if log_tot
        log = Dolo.IterationLog(;
            it=("It",Int),
            calib= ("t_calib",Float64),
            ITI= ("t_ITI", Float64),
            ergodic_dist=("t_ergodic_dist", Float64),
            discretize= ("t_discretize",Float64),
            G= ("t_G", Float64),
            A= ("t_A", Float64),
            ∂A_∂y= ("t_newton", Float64)
        )
        Dolo.initialize(log; message="Solve agent's problem") 
    else
        log = nothing
    end

    if log_∂A_∂y
        log2 = Dolo.IterationLog(;
            it=("It",Int),
            t_dp= ("t_dp",Float64),
            t_dx= ("t_dx", Float64),
            t_dμ=("t_dμ", Float64),
            t_dA= ("t_dA",Float64)
        )
        Dolo.initialize(log2; message="Details of ∂A_∂y") 
    else
        log2 = nothing
    end

    it=0

    A, dA_, dr = Ξ(y, hmodel; z=z, diff=true,  log = log, log2=log2)


    while it < n_it && maximum(abs.(A)) > toll
        ∂A_∂y = LinearMaps.LinearMap(dy -> dA_(dy; log2 = log2, it = it), N_y, N_y)
        
        if krylov
            t0 = time()
            Δy = Krylov.gmres(∂A_∂y, A* newton_adjustment_parameter)
            y = y - Δy[1]
            t_∂A_∂y = time() - t0
        else
            t0 = time()
            ∂A_∂y = convert(Matrix, ∂A_∂y)
            y = y - ∂A_∂y \  A * newton_adjustment_parameter
            t_∂A_∂y = time() - t0
        end
        it += 1

        A, dA_, dr = Ξ(y, hmodel; it=it, dr0 = dr, z=z, diff=true, log2 = log2, log = log, t_∂A_∂y = t_∂A_∂y)
    
    end

    if log_tot
        Dolo.finalize(log)
    end
    if log_∂A_∂y
        Dolo.finalize(log2)
    end

    print("y=",y, " and it=",it)
end


import Base.size
function size(lt::Dolo.LinearThing)
    return prod(Dolo.shape(lt))
end


function invert(L, r0; smaxit = 1000, tol_ν = 1e-10, krylov = true)
    if krylov
        u0 = Krylov.gmres(I-LinearMaps.LinearMap(z -> L*z,size(L)[1],size(L)[1]), r0)[1]
    else
        u0 = r0
        for i=1:smaxit
            r0 = L*r0
            u0 += r0 # supposed to be the infinite sum useful to compute an inverse
            if norm(r0)<tol_ν
                break
            end
        end
    end
    return u0
end

hmodel = Dolark.HModel("models/ayiagari.yaml")

@time solve_agent_pb(hmodel; krylov = true, log_tot = false, log_∂A_∂y = false) #0.36s























function Ξ0(y::SVector{1,Float64}, hmodel; z=SVector(0.))
    parm = hmodel.calibration[:parameters]
    p = Dolark.projection(hmodel, y, z, parm)
    r, w = p

    Dolo.set_calibration!(hmodel.agent; r=r, w=w) # update the model to account for the correct values of p

    sol_agent = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
    μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
    dmodel = Dolark.discretize(hmodel, sol_agent)
    x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])

    # computation of A = K_demand - K_offer
    return Dolark.𝒜(dmodel, μ, x, y, z; diff=false) 
end




function solve_agent_pb0(hmodel; n_it=100, toll=1e-3)

    t0= time()

    y0, z0 = hmodel.calibration[:aggregate, :exogenous]
    y = SVector(y0...)
    z = SVector(z0...)

    A = Ξ0(y, hmodel)
    it=0
    while it < n_it && maximum(abs.(A)) > toll
        ∂Ξ_∂y = FiniteDiff.finite_difference_jacobian(Y -> Ξ0(Y, hmodel), y) 
        y = y - ∂Ξ_∂y \  A
        it += 1
        A = Ξ0(y, hmodel)
        println("y=",y," and it=",it," and A=",A)
    end
    
end

hmodel = Dolark.HModel("models/ayiagari.yaml")
@time solve_agent_pb0(hmodel) # 0.7s




















# ### Tests

# z = SVector(0.)
# A, dA, dr  = Ξ(SVector(50.), hmodel; z=z, diff=true, dy= SVector(0.01))
# println("mon res:", dA )

# convert(Matrix,∂A_∂z)

# # test de dA avec FiniteDiff
# println("finitediff:",FiniteDiff.finite_difference_jacobian(Y -> Ξ(Y, hmodel; z=z)[1], SVector(50.)) * 0.01)
 
# # test avec différences finies manuelles
# parm = hmodel.calibration[:parameters]

# #test de dp —> ok
# r_p_y * SVector(0.01)
# pnew = Dolark.projection(hmodel, SVector(50.0000001), SVector(0.), parm) 
# pold = Dolark.projection(hmodel, SVector(50.), SVector(0.), parm)
# println("dp:",(pnew-pold)/0.0000001 * 0.01)

# #tests de dx et de dμ
# rold, wold  = pold
# Dolo.set_calibration!(hmodel.agent; r=rold, w=wold) # update the model to account for the correct values of p
# sol_agentold = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
# μold = Dolo.ergodic_distribution(hmodel.agent, sol_agentold)
# dmodelold = Dolark.discretize(hmodel, sol_agentold)
# xold = Dolo.MSM([sol_agentold.dr(i, dmodelold.F.s0) for i=1:length(dmodelold.F.grid.exo)])
# rnew, wnew  = pnew
# Dolo.set_calibration!(hmodel.agent; r=rnew, w=wnew) # update the model to account for the correct values of p
# sol_agentnew = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
# μnew = Dolo.ergodic_distribution(hmodel.agent, sol_agentnew)
# dmodelnew = Dolark.discretize(hmodel, sol_agentnew)
# xnew = Dolo.MSM([sol_agentnew.dr(i, dmodelnew.F.s0) for i=1:length(dmodelnew.F.grid.exo)])

# println("max dx:",maximum(((xnew-xold)/0.0000001 * 0.01).data), " and mean dx:",mean(((xnew-xold)/0.0000001 * 0.01).data))
# println("max dμ:", maximum(μnew-μold)/0.0000001 * 0.01)

# # test de dA
# println("dA:",(Dolark.𝒜(dmodelnew, μnew, xnew, SVector(50.0000001), z; diff=false) - Dolark.𝒜(dmodelold, μold, xold, SVector(50.), z; diff=false))/0.0000001 * 0.01)
