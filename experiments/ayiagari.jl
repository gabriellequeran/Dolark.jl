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
function Ξ(y::SVector{d,Float64}, hmodel; init=true, dr0 = nothing, z=SVector(0.), diff=false, smaxit=1000, tol_ν=1e-10) where d
    
    parm = hmodel.calibration[:parameters]
    p, r_p_y = Dolark.projection(hmodel, Val{(0,1)}, y, z, parm)
    r, w = p
    p = SVector{2,Float64}(p...)

    # println("time projection:",time()-t0) # ~1e-5s

    t0 = time()
    Dolo.set_calibration!(hmodel.agent; r=r, w=w) # update the model to account for the correct values of p
    println("time calib:",time()-t0) # ~1e-2s, 1e-3s

    t0 = time()
    if init # this disjunction of cases allows to speed up the time iteration algorithm when using the Newton method to find the 0 of Ξ[1]
        sol_agent = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
    else
        sol_agent = Dolo.improved_time_iteration(hmodel.agent; dr0 = dr0, verbose=false) # the second time Ξ is run, we start from dr0 as initial guess to speed up the computation
    end
    println("time ITI:",time()-t0) # ~1e-2s

    if !diff
        t0 = time()
        μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
        println("time ergodic_distribution:",time()-t0) # ~6e-3s

        t0 = time()
        dmodel = Dolark.discretize(hmodel, sol_agent) 
        println("time discretize:",time()-t0) # ~1e-3s

        x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])
    
        # computation of A = K_demand - K_offer
        t0 = time()
        A = Dolark.𝒜(dmodel, μ, x, y, z; diff=false)
        println("time A:",time()-t0) # ~3e-4s
        return A, sol_agent.dr
    end

    t0 = time()
    μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
    println("time ergodic_distribution:",time()-t0)

    t0 = time()
    dmodel = Dolark.discretize(hmodel, sol_agent)
    println("time discretize:",time()-t0)

    x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])

    t0 = time()
    μ, ∂G_∂μ, ∂G_∂x = dmodel.G(μ,x; diff=true) # here, μ is unchanged since it is already the ergodic distrib.
    println("time G", time()-t0) # ~4e-2s

    # computation of A = K_demand - K_offer and of its derivatives
    t0 = time()
    A, ∂A_∂μ, ∂A_∂x, ∂A_∂y, ∂A_∂z = Dolark.𝒜(dmodel, μ, x, y, z; diff=true)
    println("time A:",time()-t0) # ~3e-4s


    t0 = time()
    function dA_(dy)
        # computation of dp induced by dy
        dp = r_p_y * dy
        # println("time dp:",time()-t0) # 3e-4s
        # computation of dx induced by dy
        J = Dolo.df_A(dmodel.F, x, x; exo=(p,p))
        L = Dolo.df_B(dmodel.F, x, x; exo=(p,p))
        F_p1, F_p2 = Dolo.df_e(dmodel.F, x, x, p, p)
        F_p = F_p1 + F_p2
        Dolo.mult!(L, -1.0) # L : -L
        Dolo.prediv!(L, J) # L : -J\L 
        π = - J \ F_p * dp
        count = 0
        u = π
        dx = π
        for i=1:smaxit
            count +=1
            u = L*u
            dx += u # supposed to be the infinite sum useful to compute an inverse
            if norm(u)<tol_ν
                break
            end
        end
        # println("time dx:",time()-t0) # ~8e-2s

        # computation of dμ induced by dy. ∂G/∂p MUST BE ADDED AND THE CONVERGENCE MUST BE CHECKED !!!
        count=0
        U = ∂G_∂x * dx
        dμ = ∂G_∂x * dx
        for j=1:smaxit
            count +=1
            U = ∂G_∂μ * U
            dμ += U # supposed to be the infinite sum useful to compute an inverse
            if norm(U)<tol_ν
                break
            end
        end
        # println("time dμ:",time()-t0) # ~5e-3s
        # computation of dA induced by dy
        dA = convert(Matrix, ∂A_∂μ) * dμ + convert(Matrix,∂A_∂x) * dx + convert(Matrix,∂A_∂y) * dy 
        # println("time dA:",time()-t0) # ~2e-3s
        return(dA)
    end
    print("time dA_:", time()-t0)
    return A, dA_, sol_agent.dr
end


function solve_agent_pb(hmodel; n_it=100, toll=1e-3, newton_adjustment_parameter = 0.2, krylov=false) 

    y0, z0 = hmodel.calibration[:aggregate, :exogenous]
    y = SVector(y0...)
    z = SVector(z0...)
    N_y = length(y)

    it=0

    A, dA_, dr = Ξ(y, hmodel; z=z, diff=true)

    while it < n_it && maximum(abs.(A)) > toll
        t0 = time()
        ∂A_∂y = LinearMaps.LinearMap(dy -> dA_(dy), N_y, N_y)
        println("time linearmap at it ",it,": ",time()-t0)

        
        if krylov
            t0 = time()
            Δy = Krylov.gmres(∂A_∂y, A* newton_adjustment_parameter)
            println(typeof(Δy))
            y = y - Δy[1]
            println("time of newton at it ",it,": ",time()-t0)
        else
            t0 = time()
            ∂A_∂y = convert(Matrix, ∂A_∂y)
            y = y - ∂A_∂y \  A * newton_adjustment_parameter
            println("time of newton at it ",it,": ",time()-t0)
        end

        A, dA_, dr = Ξ(y, hmodel; init=false, dr0 = dr, z=z, diff=true)

        it += 1
    end

    print("y=",y, " and it=",it)
end

hmodel = Dolark.HModel("models/ayiagari.yaml")

@time solve_agent_pb(hmodel; krylov = true) #0.6s
















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
