import Dolark
import Dolo
using StaticArrays

hmodel = Dolark.HModel("models/ayiagari.yaml")

y0, z0, p = hmodel.calibration[:aggregate, :exogenous, :parameters]

m0,s0,x0,pa = hmodel.agent.calibration[:exogenous, :states, :controls, :parameters]

Dolark.projection(hmodel, y0, z0, p)

Dolark.equilibrium(hmodel, s0, x0, y0, z0, y0, z0, p)


sol_agent = Dolo.improved_time_iteration(hmodel.agent)

# sol_agent = Dolo.time_iteration(hmodel.agent)

xx0 = Dolo.MSM([sol_agent.dr(i, sol_agent.dr.grid_endo.nodes) for i=1:Dolo.n_nodes(sol_agent.dr.grid_exo)])

tab = Dolo.tabulate(hmodel.agent, sol_agent.dr, :a)

μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
# μ = μ[:]

μ = μ*0 .+ 1.0

using StaticArrays



dmodel = Dolark.discretize(hmodel, sol_agent)

x = dmodel.F.x0
y = SVector(y0...)
z = SVector(z0...)
p = Dolark.projection(hmodel, y,z,SVector(hmodel.calibration[:parameters]...))


@time Dolark.𝒜(dmodel, μ, dmodel.F.x0, y, z)



u = Dolark.Unknown(μ, p, xx0, y)



using LinearAlgebra: I
using LinearMaps


J, N_x = Dolark.Residual(dmodel, u);




using Plots

res, J = Dolark.Residual(dmodel, u);



u_ = Dolark.flatten(u)
J(u_)



M0 = convert(Matrix, J)

using FiniteDiff
M1 = FiniteDiff.finite_difference_jacobian(u->Dolark.Residual(dmodel, u; diff=false), u_)


D = abs.(M0 - M1) .>= 1e-5


spy(D)


using IterativeSolvers
import Dolark: compute_matrix

@time gmres(jj,r0)

@time δ = gmres(jj,r0; verbose=true, restart=500)

jj*δ - r0

using Plots
M = convert(Matrix, J)

spy(abs.(M).>=1e-6)


function hand_solve(jj,v)
    M = compute_matrix(jj)
    M\v
end



@time δ = gmres(jj,r0;abstol=1e-8, verbose=true, restart=100)

Δ = hand_solve(jj, r0)
@time hand_solve(jj, r0)

maximum(abs, jj*δ - r0)

jj*Δ - r0




# TODO: check that F_B(x,x) is indeed correct
@time Dolark.proto_solve_steady_state(dmodel, u);


new = proto_solve_steady_state(dmodel, u);

