

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

tab0 = Dolo.tabulate(hmodel.agent, sol_agent.dr, :a)

plot(tab0[:a], tab0[:c])

μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
# μ = μ[:]


using StaticArrays


dmodel = Dolark.discretize(hmodel, sol_agent)

# x = dmodel.F.x0
x = xx0
y = SVector(y0...)
z = SVector(z0...)
p = Dolark.projection(hmodel, y,z,SVector(hmodel.calibration[:parameters]...))


u = Dolark.Unknown(μ, p, x, y)




using LinearAlgebra: I
using LinearMaps

using Plots


res, J = Dolark.Residual(dmodel, u);

u_ = Dolark.flatten(u)
@time J*u_

r_ = Dolark.flatten(res)


# Check that jacobian is correct

@time M0 = convert(Matrix, J)

using FiniteDiff
M1 = FiniteDiff.finite_difference_jacobian(u->Dolark.Residual(dmodel, u; diff=false), u_) # ; relstep=1e-10)
M1_ = jacobian(forward_fdm(5, 1), u->Dolark.Residual(dmodel, u; diff=false), u_)[1]  # more precise


DD = abs.(M0 - M1_)
D = (DD .>= 1e-8)

D[1,1]=1
D[end,end]=1

spy(D)

# There are non zero values for F_p, i.e. DD[303:602,301:302]

# M0[1,1:90] .= 1.0
# u_[1] = 0


k1 = xx0
k2 = xx0
kp = p

# dmodel.F(k1, k2, p, p)

R =  dmodel.F(k1, k2)
N_x = length(k1.data)*length(k1.data[1])
N_p = length(p)

L_A = Dolo.df_A(dmodel.F, k1, k2)
L_B = Dolo.df_B(dmodel.F, k1, k2)
L_p1, L_p2 = Dolo.df_e(dmodel.F, xx0, xx0, p,p)
J_A = LinearMaps.LinearMap(u->L_A*u, N_x, N_x)
J_B = LinearMaps.LinearMap(u->L_B*u, N_x, N_x)
J_p1 = LinearMaps.LinearMap(u->L_p1*u, N_x, N_p)
J_p2 = LinearMaps.LinearMap(u->L_p2*u, N_x, N_p)


import LinearMaps
M_A = convert(Matrix,(J_A))
M_B = convert(Matrix,(J_B))
M_p1 = convert(Matrix,(J_p1))
M_p2 = convert(Matrix,(J_p2))


ff = u->flatten(dmodel.F(unflatten(u, k1), k2))
M1 = FiniteDiff.finite_difference_jacobian(ff, flatten(k1); absstep=1e-8)
M1_ = jacobian(central_fdm(5, 1), ff, flatten(k1))[1] # this is more precise


ff = u->flatten(dmodel.F(k1, unflatten(u, k2)))
M2 = FiniteDiff.finite_difference_jacobian(ff, flatten(k2)) # ; relstep=1e-10)
M2_ = jacobian(central_fdm(5, 1), ff, flatten(k2))[1]


ff = u->flatten(dmodel.F(k1, k2; exo=(u,p)))
Mp1 = FiniteDiff.finite_difference_jacobian(ff, p) # ; relstep=1e-10)
Mp1_ = jacobian(forward_fdm(5, 1), ff, p)[1]

ff = u->flatten(dmodel.F(k1, k2; exo=(p,u)))
Mp2 = FiniteDiff.finite_difference_jacobian(ff, p) # ; relstep=1e-10)
Mp2_ = jacobian(central_fdm(5, 1), ff, p)[1]


using FiniteDifferences


maximum(abs,M_A - M1_)   # 1e-7
maximum(abs,M_B - M2_)  ## very small
maximum(abs,M_p1 - Mp1)  # 1e-6
maximum(abs,M_p2 - Mp2)  # 1e-6
# there is probably something wrong in the chain derivatives inside Euler (CHECK)


# try to solve the system

@time us = Dolark.proto_solve_steady_state(dmodel, u; numdiff=true, use_blas=true, maxit=5);   # this "seems" to work (it is slow)

# check the solution
dr = deepcopy(dmodel.F.dr.dr)
Dolo.set_values!(dr, us.x)

tab = Dolo.tabulate(hmodel.agent, dr, :a)


# plot!(pl, tab[:a], tab[:i])
# plot!(pl, tab[:a], tab[:a])
pl = plot(tab[:a], tab[:c])
plot!(pl, tab0[:a], tab0[:c]) # it looks reasonable



### now we want to use fast jacobian calculation
sol = Dolark.proto_solve_steady_state(dmodel, u; numdiff=false, use_blas=true, maxit=5) #this doesn't work so well


@time new = Dolark.proto_solve_steady_state(dmodel, u);



