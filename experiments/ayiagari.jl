using StaticArrays
import Dolark
import Dolo

function set_calibration_dmodel!(dmodel::Dolark.DModel, key::Symbol, value)
    Dolo.set_calibration!(dmodel.hmodel.agent, key, value)
end


function solve_agent_pb(hmodel; n_it=30, toll=1e-8)

    it = 0

    # initialization

    y0, z0, p0 = hmodel.calibration[:aggregate, :exogenous, :parameters]
    y = SVector(y0...)
    z = SVector(z0...)
    p = Dolark.projection(hmodel, y, z, p0) # warning: p and p0 are not the same kind of arguments. p regroups r and w whereas p0 is a list of constant parameters
    r, w = p

    Dolo.set_calibration!(hmodel.agent; r=r, w=w)
    sol_agent = Dolo.improved_time_iteration(hmodel.agent)
    μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
    dmodel = Dolark.discretize(hmodel, sol_agent)

    x = dmodel.F.x0

    # computation of A = K_demand - K_offer, and of its derivatives
    A, R_A_mu, R_A_x, R_A_y, R_A_z = Dolark.𝒜(dmodel, μ, x, y, z; diff=true)

    while it < n_it && abs(A[1]) > toll 
        y = y - convert(Matrix, R_A_y) \ A # Newton's method to update y. 
        p = Dolark.projection(dmodel.hmodel, y, z, p0)
        r, w = p

        print("abs(A[1])=",abs(A[1])," and y=",y," and ","r=",r," and w=",w,";    ")

        Dolo.set_calibration!(dmodel.hmodel.agent; r=r, w=w) #updating the agent's model
        sol_agent = Dolo.improved_time_iteration(dmodel.hmodel.agent; verbose=false)
        μ = Dolo.ergodic_distribution(dmodel.hmodel.agent, sol_agent)
        x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])

        A, R_A_mu, R_A_x, R_A_y, R_A_z = Dolark.𝒜(dmodel, μ, x, y, z; diff=true)

        it += 1
    end
    print("Finally: y=",y," and A=",A, " and it =", it) 
end


hmodel = Dolark.HModel("models/ayiagari.yaml")
solve_agent_pb(hmodel) #At least, it works the third time.














# sol_agent = Dolo.improved_time_iteration(hmodel.agent)
# μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
# dmodel = Dolark.discretize(hmodel, sol_agent)

# y0, z0, p0 = hmodel.calibration[:aggregate, :exogenous, :parameters]
# y = SVector(y0...)
# z = SVector(z0...)

# A, R_A_mu, R_A_x, R_A_y, R_A_z = Dolark.𝒜(dmodel, μ, dmodel.F.x0, y, z; diff=true)

# set_calibration!(hmodel, :r, 0.5)

# dmodel.F.x0

# y0

# z0

# Dolark.projection(dmodel.hmodel, y, z, p0)

# p0
# p
