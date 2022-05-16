using FiniteDiff
using Dolo
import Dolark

# @testset "Test the 𝒜 derivatives w.r.t. x, μ, z and y" begin

    hmodel = Dolark.HModel("models/ayiagari.yaml")

    sol_agent = Dolo.improved_time_iteration(hmodel.agent)
    μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
    dmodel = Dolark.discretize(hmodel, sol_agent)

    y0, z0 = hmodel.calibration[:aggregate, :exogenous]
    y = SVector(y0...)
    z = SVector(z0...)

    A, R_A_mu, R_A_x, R_A_y, R_A_z = Dolark.𝒜(dmodel, μ, dmodel.F.x0, y, z; diff=true)
    

    Jμ_exact = convert(Matrix, R_A_mu)
    Jμ_num = FiniteDiff.finite_difference_jacobian(mu -> Dolark.𝒜(dmodel, mu, dmodel.F.x0, y, z), μ)

    Jx_exact = convert(Matrix, R_A_x)
    # Jx_num = []
    # for i in 1:length(dmodel.F.x0.data)
    #     x_forward = cat(dmodel.F.x0.data...,dims=1)
    #     x_forward[1] += 1e-8
    #     x_forward = Dolo.MSM(copy(reinterpret(SVector{n_x, Float64}, x_forward)), dmodel.F.x0.sizes)
    #     append!(Jx_num, (Dolark.𝒜(dmodel, μ, x_forward, y, z)-Dolark.𝒜(dmodel, μ, dmodel.F.x0, y, z))/1e-8)
    # end
    # Jx_num = [Float64(i) for i in Jx_num] 
    
    Jx_num = FiniteDiff.finite_difference_jacobian(X0 -> Dolark.𝒜(dmodel, μ, X0, y, z), dmodel.F.x0)
    
eltype(μ)
eltype(dmodel.F.x0)
typeof(dmodel.F.x0)



    R_A_x

    Jy_exact = convert(Matrix, R_A_y)
    Jy_num = FiniteDiff.finite_difference_jacobian(Y-> Dolark.𝒜(dmodel, μ, dmodel.F.x0, Y, z), y)
    
    Jz_exact = convert(Matrix, R_A_z)
    Jz_num = FiniteDiff.finite_difference_jacobian(Z-> Dolark.𝒜(dmodel, μ, dmodel.F.x0, y, Z), z)

    maximum(abs, Jμ_num - Jμ_exact) < 1e-8 #@assert 

    maximum(abs, Jx_num - Jx_exact) < 1e-5 #@assert 

    maximum(abs, Jy_num - Jy_exact) < 1e-8 #@assert

    maximum(abs, Jz_num - Jz_exact) < 1e-8 #@assert




# end