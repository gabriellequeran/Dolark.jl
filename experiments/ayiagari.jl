using StaticArrays
import Dolark
import Dolo


function solve_agent_pb(hmodel; n_it=100, toll=1e-8)

    it = 0

    # initialization
    y0, z0, parm = hmodel.calibration[:aggregate, :exogenous, :parameters]
    y = SVector(y0...)
    z = SVector(z0...)
    p = Dolark.projection(hmodel, y, z, parm) 
    r, w = p

    Dolo.set_calibration!(hmodel.agent; r=r, w=w)
    print(" initialization: y=",y," and ","r=",hmodel.agent.calibration.flat[:r]," and w=",hmodel.agent.calibration.flat[:w],";    ")
    sol_agent = Dolo.improved_time_iteration(hmodel.agent)
    μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
    dmodel = Dolark.discretize(hmodel, sol_agent)

    x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)]) #dmodel.F.x0

    # computation of A = K_demand - K_offer, and of its derivatives
    A, R_A_mu, R_A_x, R_A_y, R_A_z = Dolark.𝒜(dmodel, μ, x, y, z; diff=true)

    while it < n_it && abs(A[1]) > toll 
        y = y - convert(Matrix, R_A_y) \ A # Newton's method to update y. 

        p = Dolark.projection(dmodel.hmodel, y, z, parm)
        r, w = p

        print("A=",A," and y=",y," and ","r=",r," and w=",w,";    ")
        # y_new = y - convert(Matrix, R_A_y) \ A # Newton's method to update y. 

        # p = Dolark.projection(dmodel.hmodel, y_new, z, parm)
        # r, w = p

        # print("A=",A," and y=",y_new," and ","r=",r," and w=",w,";    ")

        # if isnan(r)
        #     y = (y+99*y_new) ./100
        #     p = Dolark.projection(dmodel.hmodel, y, z, parm)
        #     r, w = p
        #     print("yes. Now: y=",y," and ","r=",r," and w=",w,";    ")
        # else
        #     y = y_new
        # end

        Dolo.set_calibration!(hmodel.agent; r=r, w=w) #updating the agent's model
        sol_agent = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
        dmodel = Dolark.discretize(hmodel, sol_agent)
        μ = Dolo.ergodic_distribution(dmodel.hmodel.agent, sol_agent)
        x = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])

        A, R_A_mu, R_A_x, R_A_y, R_A_z = Dolark.𝒜(dmodel, μ, x, y, z; diff=true)

        it += 1
    end
    print("Finally: y=",y," and A=",A, " and it =", it) 
end


hmodel = Dolark.HModel("models/ayiagari.yaml")
solve_agent_pb(hmodel) #At least, it works the third time.
