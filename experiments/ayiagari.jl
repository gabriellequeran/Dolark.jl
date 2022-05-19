using StaticArrays
import Dolark
import Dolo


function Ξ(y::SVector{1,Float64}, hmodel; z=SVector(0.)) # ~ 0.3s
    parm = hmodel.calibration[:parameters]
    p = Dolark.projection(hmodel, y, z, parm)
    r, w = p

    Dolo.set_calibration!(hmodel.agent; r=r, w=w) # update the model to account for the correct values of p

    println("r=",r,"; w=",w)
    #println(hmodel.source)
    println(hmodel.calibration)
    println(hmodel.exogenous)
    println(hmodel.equations)
    println(hmodel.agent.calibration)

    sol_agent = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
    μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
    dmodel = Dolark.discretize(hmodel, sol_agent)
    x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])

    # computation of A = K_demand - K_offer
    return Dolark.𝒜(dmodel, μ, x, y, z; diff=false) 
end

hmodel = Dolark.HModel("models/ayiagari.yaml")
Ξ(SVector(50.),hmodel)[1]

function Ξ2(y::SVector{1,Float64}, hmodel; z=SVector(0.)) # Ξ2 applique 2 fois les opérations de Ξ et cherche à comprendre ce qui change
    parm = hmodel.calibration[:parameters]
    p = Dolark.projection(hmodel, y, z, parm)
    r, w = p

    Dolo.set_calibration!(hmodel.agent; r=r, w=w) # update the model to account for the correct values of p
    sol_agent = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
    μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
    dmodel = Dolark.discretize(hmodel, sol_agent)
    x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])

    hmodel1 = hmodel
    sol_agent1 = sol_agent
    dmodel1 = dmodel
    μ1 = μ
    x1 = x

    parm = hmodel.calibration[:parameters]
    p = Dolark.projection(hmodel, y, z, parm)
    r, w = p
    Dolo.set_calibration!(hmodel.agent; r=r, w=w) # update the model to account for the correct values of p
    sol_agent2 = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
    μ2 = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
    dmodel2 = Dolark.discretize(hmodel, sol_agent)
    x2 = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])

    println(hmodel1==hmodel, sol_agent1==sol_agent2, dmodel1==dmodel2, μ1==μ2, x1==x2)
    println("A1:",Dolark.𝒜(dmodel1, μ1, x1, y, z; diff=false)," A2:",Dolark.𝒜(dmodel2, μ2, x2, y, z; diff=false))
end

hmodel = Dolark.HModel("models/ayiagari.yaml")
Ξ2(SVector(50.), hmodel) 











function solve_agent_pb(hmodel; n_it=100, toll=1e-3)
    y0, z0 = hmodel.calibration[:aggregate, :exogenous]
    y = SVector(y0...)
    z = SVector(z0...)

    A = Ξ(y, hmodel)
    it=0
    println("y=",y," and it=",it," and A=",A,"     ")
    while it < n_it && maximum(abs.(A)) > toll
        ∂Ξ_∂y = FiniteDiff.finite_difference_jacobian(Y -> Ξ(Y, hmodel), y) 
        y = y - ∂Ξ_∂y \  A
        it += 1
        println("y=",y," and A not yet calculated =", Ξ(y, hmodel))
        A = Ξ(y, hmodel)
        println("y=",y," and it=",it," and A=",A,"     ")
    end
    
end

hmodel = Dolark.HModel("models/ayiagari.yaml")
solve_agent_pb(hmodel)





using Plots
p1 = plot([k for k in 40:80], [Ξ(SVector(k+0.),hmodel)[1] for k in 40:80], label="Ξ")
p2 = plot([k for k in 40:80], [FiniteDiff.finite_difference_jacobian(y -> Ξ(y, hmodel), SVector(k+0.))[1] for k in 40:80], label="∂Ξ_∂y")
plot(p1,p2, layout =(2,1))






# Looking at proto_solve_steady_state
y0, z0 = hmodel.calibration[:aggregate, :exogenous]
y = SVector(y0...)
z = SVector(z0...)
parm = hmodel.calibration[:parameters]
p = Dolark.projection(hmodel, y, z, parm)
r, w = p
Dolo.set_calibration!(hmodel.agent; r=r, w=w) # update the model to account for the correct values of p
sol_agent = Dolo.improved_time_iteration(hmodel.agent)
μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
dmodel = Dolark.discretize(hmodel, sol_agent)
x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])

u0 = Dolark.Unknown(μ,p,x,y)
Dolark.proto_solve_steady_state(dmodel, u0; numdiff=false, use_blas=true, maxit=10, toll=1e-5)
