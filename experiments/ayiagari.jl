using StaticArrays
import Dolark
import Dolo
using FiniteDiff
using LinearMaps
using LinearAlgebra
using Statistics







function Ξ(y::SVector{1,Float64}, hmodel; init=true, dr0 = nothing, z=SVector(0.))
    parm = hmodel.calibration[:parameters]
    p, r_p_y = Dolark.projection(hmodel, Val{(0,1)}, y, z, parm)
    r, w = p

    Dolo.set_calibration!(hmodel.agent; r=r, w=w) # update the model to account for the correct values of p

    if init 
        sol_agent = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
    else
        sol_agent = Dolo.improved_time_iteration(hmodel.agent; dr0 = dr0, verbose=false)
    end

    μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
    dmodel = Dolark.discretize(hmodel, sol_agent)
    x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])
    μ, ∂G_∂μ, ∂G_∂x = dmodel.G(μ,x; diff=true)

    # computation of A = K_demand - K_offer
    A, R_A_mu, R_A_x, R_A_y, R_A_z = Dolark.𝒜(dmodel, μ, x, y, z; diff=true)
    return A, x, p, R_A_mu, ∂G_∂x, dmodel.F, r_p_y, sol_agent.dr
end

hmodel = Dolark.HModel("models/ayiagari.yaml")
A, x, p, ∂A_∂μ, ∂G_∂x, F, r_p_y, dr = Ξ(SVector(2.), hmodel)
A, x, p, ∂A_∂μ, ∂G_∂x, F, r_p_y, dr = Ξ(SVector(2.), hmodel; init = false, dr0 = dr)



function ∂Ξ_∂y_x_dy(x, p::SVector{np,Float64}, dy::SVector{ny,Float64}, ∂A_∂μ, ∂G_∂x, F, r_p_y; tol_ν = 1e-10, smaxit = 3) where ny where np
    dp = r_p_y * dy
    println("p:",p," and dp:",dp)

    F(x,x;exo=(p,p), set_future=true)
    J = Dolo.df_A(F, x, x; exo=(p,p))
    L = Dolo.df_B(F, x, x; exo=(p,p))
    F_p1, F_p2 = Dolo.df_e(F, x, x, p, p)
    F_p = F_p1 + F_p2
    Dolo.mult!(L, -1.0) # L : -L
    Dolo.prediv!(L, J) # L : -J\L 
    π = - J \ F_p * dp
    count = 0
    u = π
    δ = π
    for i=1:smaxit
        count +=1
        u = L*u
        δ += u #supposed to be the infinite sum useful to compute an inverse
        if norm(u)<tol_ν
            break
        end
    end
    dx = δ 
    #println("max de x:",maximum(x.data)," and max de dx:",maximum(dx.data))
    println("max de dx:",maximum(dx)," and mean x:", mean(dx))

    dμ = convert(Matrix, ∂G_∂x) * dx
    println("max de dμ:",maximum(dμ))

    ∂Ξ_∂y_x_dy = ∂A_∂μ * dμ

    return [∂Ξ_∂y_x_dy]
end

A, x, p, ∂A_∂μ, ∂G_∂x, F, r_p_y, dr = Ξ(SVector(2.), hmodel)
println("mon res:",∂Ξ_∂y_x_dy(x, SVector(p...), SVector(0.01), ∂A_∂μ, ∂G_∂x, F, r_p_y) )

# test de dA avec FiniteDiff
println("finitediff:",FiniteDiff.finite_difference_jacobian(Y -> Ξ(Y, hmodel)[1], SVector(2.)) * 0.01)


# test avec différences finies manuelles
parm = hmodel.calibration[:parameters]

#test de dp —> ok
r_p_y * SVector(0.01)
pnew = Dolark.projection(hmodel, SVector(2.0000001), SVector(0.), parm) 
pold = Dolark.projection(hmodel, SVector(2.), SVector(0.), parm)
println("dp:",(pnew-pold)/0.0000001 * 0.01)

#tests de dx et de dμ
rold, wold  = pold
Dolo.set_calibration!(hmodel.agent; r=rold, w=wold) # update the model to account for the correct values of p
sol_agentold = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
μold = Dolo.ergodic_distribution(hmodel.agent, sol_agentold)
dmodelold = Dolark.discretize(hmodel, sol_agentold)
xold = Dolo.MSM([sol_agentold.dr(i, dmodelold.F.s0) for i=1:length(dmodelold.F.grid.exo)])
rnew, wnew  = pnew
Dolo.set_calibration!(hmodel.agent; r=rnew, w=wnew) # update the model to account for the correct values of p
sol_agentnew = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
μnew = Dolo.ergodic_distribution(hmodel.agent, sol_agentnew)
dmodelnew = Dolark.discretize(hmodel, sol_agentnew)
xnew = Dolo.MSM([sol_agentnew.dr(i, dmodelnew.F.s0) for i=1:length(dmodelnew.F.grid.exo)])

println("max dx:",maximum(((xnew-xold)/0.0000001 * 0.01).data), " and mean dx:",mean(((xnew-xold)/0.0000001 * 0.01).data))
println("max dμ:", maximum(μnew-μold)/0.0000001 * 0.01)

# test de dA
println("dA:",(Dolark.𝒜(dmodelnew, μnew, xnew, SVector(2.0000001), z; diff=false) - Dolark.𝒜(dmodelold, μold, xold, SVector(2.), z; diff=false))/0.0000001 * 0.01)













function solve_agent_pb(hmodel; n_it=100, toll=1e-3)

    t0= time()

    y0, z0 = hmodel.calibration[:aggregate, :exogenous]
    y = SVector(y0...)
    z = SVector(z0...)
    N_y = length(y)

    A, x, p, ∂A_∂μ, ∂G_∂x, F, r_p_y, dr = Ξ(y, hmodel; z=z)
    it=0
    println("y=",y," and it=",it," and A=",A," and time=",time()-t0,"s     ")
    while it < n_it && maximum(abs.(A)) > toll

        ∂Ξ_∂y = LinearMaps.LinearMap(dy -> ∂Ξ_∂y_x_dy(x, SVector(p...), SVector(dy...), ∂A_∂μ, ∂G_∂x, F, r_p_y), N_y, N_y)
        ∂Ξ_∂y = convert(Matrix, ∂Ξ_∂y)
        y = y - ∂Ξ_∂y \  A
        it += 1
        A, x, p, ∂A_∂μ, ∂G_∂x, F, r_p_y, dr = Ξ(y, hmodel; init=false, dr0=dr, z=z)
        println("y=",y," and it=",it," and A=",A," and time=",time()-t0,"s     ")
    end
    
end

solve_agent_pb(hmodel)


















# function Ξ(y::SVector{1,Float64}, hmodel; z=SVector(0.))
#     parm = hmodel.calibration[:parameters]
#     p = Dolark.projection(hmodel, y, z, parm)
#     r, w = p

#     Dolo.set_calibration!(hmodel.agent; r=r, w=w) # update the model to account for the correct values of p

#     sol_agent = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
#     μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
#     dmodel = Dolark.discretize(hmodel, sol_agent)
#     x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])

#     # computation of A = K_demand - K_offer
#     return Dolark.𝒜(dmodel, μ, x, y, z; diff=false) 
# end




# function solve_agent_pb(hmodel; n_it=100, toll=1e-3)

#     t0= time()

#     y0, z0 = hmodel.calibration[:aggregate, :exogenous]
#     y = SVector(y0...)
#     z = SVector(z0...)

#     A = Ξ(y, hmodel)
#     it=0
#     println("y=",y," and it=",it," and A=",A," and time=",time()-t0,"s     ")
#     while it < n_it && maximum(abs.(A)) > toll
#         ∂Ξ_∂y = FiniteDiff.finite_difference_jacobian(Y -> Ξ(Y, hmodel), y) 
#         y = y - ∂Ξ_∂y \  A
#         it += 1
#         A = Ξ(y, hmodel)
#         println("y=",y," and it=",it," and A=",A," and time=",time()-t0,"s     ")
#     end
    
# end

# hmodel = Dolark.HModel("models/ayiagari.yaml")
# solve_agent_pb(hmodel)





# t0= time()

#     y0, z0 = hmodel.calibration[:aggregate, :exogenous]
#     y = SVector(y0...)
#     z = SVector(z0...)
#     N_y = length(y)

#     A, x, p, ∂A_∂μ, ∂G_∂x, F, r_p_y, dr = Ξ(y, hmodel; z=z)
#     it=0
#     # println("y=",y," and it=",it," and A=",A," and time=",time()-t0,"s     ")
#     # while it < 20 && maximum(abs.(A)) > 1e-4

#         ∂Ξ_∂y = LinearMaps.LinearMap(dy -> ∂Ξ_∂y_x_dy(x, SVector(p...), SVector(dy...), ∂A_∂μ, ∂G_∂x, F, r_p_y), N_y, N_y)
#         ∂Ξ_∂y = convert(Matrix, ∂Ξ_∂y)
#         y = y - ∂Ξ_∂y \  A
#         it += 1
#         A, x, p, ∂A_∂μ, ∂G_∂x, F, r_p_y, dr = Ξ(SVector(2.), hmodel; init=false, dr0=dr, z=z)
#         println("y=",y," and it=",it," and A=",A," and time=",time()-t0,"s     ")
#     # end





















# p = SVector(p...)
# dy = SVector(0.01)

#     dp = r_p_y * dy

#     J = Dolo.df_A(F, x, x; exo=(p,p))
#     L = Dolo.df_B(F, x, x; exo=(p,p))
#     F_p1, F_p2 = Dolo.df_e(F, x, x, p, p)
#     F_p = F_p1 + F_p2
#     Dolo.mult!(L, -1.0) # L : -L
#     Dolo.prediv!(L, J) # L : -J\L 
#     π = J \ A #Dolo.MSM(1. * SMatrix(I))
#     count = 0
#     u = π
#     δ = π
#     for i=1:3
#         count +=1
#         u = L*u
#         δ += u #supposed to be the infinite sum useful to compute an inverse
#         if norm(u)<1e-8
#             break
#         end
#     end
#     mult!(δ, -1.0)
#     dx = δ * (J \ F_p) * dp

#     dμ = ∂G_∂x * dx

#     ∂Ξ_∂y_x_dy = ∂A_∂μ * dμ


# using Plots
# p1 = plot([k for k in 40:80], [Ξ(SVector(k+0.),hmodel)[1] for k in 40:80], label="Ξ")
# p2 = plot([k for k in 40:80], [FiniteDiff.finite_difference_jacobian(y -> Ξ(y, hmodel), SVector(k+0.))[1] for k in 40:80], label="∂Ξ_∂y")
# plot(p1,p2, layout =(2,1))


# sol_agent = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
# Dolo.improved_time_iteration(hmodel.agent; dr0= sol_agent.dr, verbose=false)



# # # Looking at proto_solve_steady_state
# # y0, z0 = hmodel.calibration[:aggregate, :exogenous]
# # y = SVector(y0...)
# # z = SVector(z0...)
# # parm = hmodel.calibration[:parameters]
# # p = Dolark.projection(hmodel, y, z, parm)
# # r, w = p
# # Dolo.set_calibration!(hmodel.agent; r=r, w=w) # update the model to account for the correct values of p
# # sol_agent = Dolo.improved_time_iteration(hmodel.agent)
# # μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
# # dmodel = Dolark.discretize(hmodel, sol_agent)
# # x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])

# # u0 = Dolark.Unknown(μ,p,x,y)
# # Dolark.proto_solve_steady_state(dmodel, u0; numdiff=false, use_blas=true, maxit=10, toll=1e-5)
