import Dolark
import Dolo
using StaticArrays

# We need YAML.jl from the master branch

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


using StaticArrays


dmodel = Dolark.discretize(hmodel, sol_agent)

# x = dmodel.F.x0
x = xx0
y = SVector(y0...)
z = SVector(z0...)
p = Dolark.projection(hmodel, y,z,SVector(hmodel.calibration[:parameters]...))


@time Dolark.𝒜(dmodel, μ, dmodel.F.x0, y, z)



u = Dolark.Unknown(μ, p, xx0, y)



using LinearAlgebra: I
using LinearMaps


J, N_x = Dolark.Residual(dmodel, u);




using Plots

res, J, X = Dolark.Residual(dmodel, u);

u_ = Dolark.flatten(u)
J(u_)

r_ = Dolark.flatten(res)

M0 = convert(Matrix, J)

using FiniteDiff
M1 = FiniteDiff.finite_difference_jacobian(u->Dolark.Residual(dmodel, u; diff=false), u_)


D = abs.(M0 - M1) .>= 1e-6

spy(D)

# M0[1,1:90] .= 1.0
# u_[1] = 0

X*u_

using IterativeSolvers
function power_method(X)
    u0 = rand(603)
    λ = maximum(abs, u0)
    u0 = u0 / λ
    for i =1:1000
        u1 = X*u0
        λ = maximum(abs, u1)
        u0 = u1/λ
    end
    return λ
end


pow(X)
M = convert(Matrix, X)
using LinearAlgebra
evs = eigvals(M)

evs = abs.(evs)

sol = M0 \ r_

sol1 = M1 \ r_

using Plots

spy(D)


Dolark.proto_solve_steady_state(dmodel, u)


using IterativeSolvers

sol_g = gmres(J, r_, abstol=1e-10)

J * sol_g - r_


@time gmres(jj,r0)

@time δ = gmres(J,r_; verbose=true, restart=500)

J*δ - r_

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


@time new = Dolark.proto_solve_steady_state(dmodel, u);










hmodel = Dolark.HModel("models/ayiagari.yaml")


hmodel.agent.domain

# sol = Dolo.improved_time_iteration(hmodel.agent)
# hmodel.agent.domain

Dolo.set_calibration!(hmodel.agent; r=0.001, w=3)

dmodel = Dolark.discretize(hmodel, sol)
hmodel.agent.domain

sol = Dolo.improved_time_iteration(dmodel.hmodel.agent)

using StaticArrays

function fun(m, y, sol0)
    agent = m.hmodel.agent

    y_ = SVector(y...)
    z_ = m.hmodel.calibration[:exogenous]
    p_ = m.hmodel.calibration[:parameters]
    # Dolo.set_calibration!(m.hmodel.agent; r=r, w=w)
    p = Dolark.projection(m.hmodel, y_, z_, p_)
    r,w = p
    # Dolo.set_calibration!(m.hmodel.agent; r=r, w=w)
    Dolo.set_calibration!(m.hmodel.agent; r=r, w=w)

    # soll = Dolo.improved_time_iteration(agent;dr0= sol0.dr, verbose=false)
    soll = Dolo.improved_time_iteration(agent; verbose=false)
    μ = Dolo.ergodic_distribution(agent, soll)
    x = Dolo.MSM([soll.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])
    res = Dolark.𝒜(m, μ, x, y_, z_)
    return res

end

xvec = range(40, 65;length= 100)
yvec = [fun(dmodel, [k], sol_agent) for k in xvec ]

using Plots
plot(xvec, yvec)
