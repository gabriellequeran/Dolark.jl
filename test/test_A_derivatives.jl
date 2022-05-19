using FiniteDiff
using StaticArrays
import Dolark
import Dolo

@testset "Test 𝒜's derivatives w.r.t. x, μ, z and y" begin

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


    flatten(u::Dolo.MSM) = cat(u.data...;dims=1)                              # converts an MSM into a float64 vector
    unflatten(u, x) = Dolo.MSM([reinterpret(eltype(x.data),u)...], x.sizes)   # converts a float64 vector into an MSM


    x0_flat = flatten(dmodel.F.x0)

    Jx_num = FiniteDiff.finite_difference_jacobian(X0 -> Dolark.𝒜(dmodel, μ, unflatten(X0, x0), y, z), x0_flat)


    Jy_exact = convert(Matrix, R_A_y)
    Jy_num = FiniteDiff.finite_difference_jacobian(Y-> Dolark.𝒜(dmodel, μ, dmodel.F.x0, Y, z), y)
    
    Jz_exact = convert(Matrix, R_A_z)
    Jz_num = FiniteDiff.finite_difference_jacobian(Z-> Dolark.𝒜(dmodel, μ, dmodel.F.x0, y, Z), z)



    @assert maximum(abs, Jμ_num - Jμ_exact) < 1e-8 

    @assert maximum(abs, Jx_num - Jx_exact) < 1e-8 

    @assert maximum(abs, Jy_num - Jy_exact) < 1e-8 

    @assert maximum(abs, Jz_num - Jz_exact) < 1e-8 






end