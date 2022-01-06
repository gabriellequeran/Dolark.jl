
function ℱ(dmodel, μ0, x0, y0, z0)

end
using StaticArrays

function 𝒜(dmodel, μ0, x0, y0::SVector{d, Float64}, z0; diff=false) where d

    N = length(μ0)
    
    μ = μ0[:]

    s0 = repeat(dmodel.F.s0, length(x0.views))

    zvec = [z0 for n=1:N]
    yvec = [y0 for n=1:N]

    parms = SVector(dmodel.hmodel.calibration[:parameters]...)

    res = Dolark.equilibrium(dmodel.hmodel, s0, x0.data, yvec, zvec, yvec, zvec, parms)
    res = reinterpret(Float64, res)



    if diff==false

        return sum(μ .* res)
    
    else

        A = sum(μ .* res)

        e_x0, e_y0, e_z0, e_y1, e_z1 = Dolark.equilibrium(dmodel.hmodel, Val{(2,3,4,5,6)}, s0, x0.data, yvec, zvec, yvec, zvec, parms)
        
        n_x = size(e_x0[1],2)
        n_y = size(e_y0[1],2)
        n_z = size(e_z0[1],2)

        A_mu = d_mu -> sum(d_mu .* res)
        A_x = d_x -> sum(μ .* (e_x0 .* reinterpret(SVector{n_x, Float64}, d_x)))
        A_y = d_y -> sum(μ .* [(e_y0 .+ e_y1)[n]*(SVector{n_y, Float64}(d_y...)) for n=1:N] )
        A_z = d_z -> sum(μ .* [(e_z0 .+ e_z1)[n]*(SVector{n_z, Float64}(d_z...)) for n=1:N] )

        # A_mu = d_mu -> sum(d_mu .* res)
        # A_x = d_x -> sum(μ .* e_x0 .* d_x.data)
        # A_y = d_y -> sum(μ .* [(e_y0 .+ e_y1)[n]*d_y for n=1:N] )
        # A_z = d_z -> sum(μ .* [(e_z0 .+ e_z1)[n]*d_z for n=1:N] )

        n_y = length(e_x0[1])
        N_μ = length(μ)
        N_x = length(x0.data)
        n_z = size(e_z1[1],2)
    
        R_A_mu = LinearMaps.LinearMap(A_mu, n_y, N_μ)
        R_A_x = LinearMaps.LinearMap(A_x, n_y, N_x)
        R_A_y = LinearMaps.LinearMap(A_y, n_y, n_y)
        R_A_z = LinearMaps.LinearMap(A_z, n_y, n_z)

        return A, R_A_mu, R_A_x, R_A_y, R_A_z

    end

end

using LinearAlgebra: I


using LinearMaps


using LinearAlgebra: I

function Residual(dmodel, u::Unknown)

    z = SVector(dmodel.hmodel.calibration[:exogenous]...)
    p_ = SVector(dmodel.hmodel.calibration[:parameters]...)

    μ = u.μ[:]
    x = u.x
    y = u.y
    p = u.p

    N_μ = length(μ)
    N_x = length(x.data)*length(x.data[1])
    n_p = length(p)
    n_y = length(y)
    n_z = length(z)

    F = dmodel.F
    G = dmodel.G


    # quick computations if we don't compute derivatives
    # r_μ = μ - G(u.μ, u.x)
    r_F = F(x, x; exo=(p,p), set_future=true) # this is not equal to the normalized version
    # r_A = 𝒜(dmodel, μ, x, y, z)
    # r_p = Dolark.projection(dmodel.hmodel, y, z, p_)


    μ1, G_mu, G_x = G(u.μ, u.x, diff=true)
   
    r_μ = μ - μ1

    r_F = F(x, x; exo=(p,p), set_future=true) # this is not equal to the normalized version
    JJ = Dolo.df_A(F, x,x; exo=(p,p))
    L = Dolo.df_B(F, x,x; exo=(p,p))
    F_p1, F_p2 = Dolo.df_e(F, x, x, p,p)
    F_p = F_p1 + F_p2 # that one looks wrong

    Dolo.prediv!(L, JJ)
    r_F = JJ\r_F
    F_p = JJ\F_p

    Ft_x = LinearMaps.LinearMap(z->JJ*z, N_x, N_x)
    # Ft_x = LinearMaps.LinearMap(z->z, N_x, N_x)
    Ft_xx = LinearMaps.LinearMap(z->L*z, N_x, N_x)
    Ft_p = LinearMaps.LinearMap(z->F_p*z, N_x, n_p)


    r_p, r_p_y, r_p_z = Dolark.projection(dmodel.hmodel, Val{(0,1,2)}, y, z, p_)

    R_p_y = LinearMaps.LinearMap(z->r_p_y*z, n_p, n_y)
    R_p_z = LinearMaps.LinearMap(z->r_p_z*z, n_p, n_z)


    r_A, r_A_mu, r_A_x, r_A_y, r_A_z = 𝒜(dmodel, μ, x, y, z; diff=true)

    u = Dolark.Unknown(μ - μ1, p - r_p, r_F, r_A)

    J = [ I-G_mu                 zeros(N_μ, n_p)        -G_x               zeros(N_μ, n_y)         ; # μ
          zeros(n_p, N_μ)         I                     zeros(n_p, N_x)     -R_p_y        # p;
          zeros(N_x, N_μ)        Ft_p                  Ft_x+Ft_xx          zeros(N_x, n_y)         ; # x
          r_A_mu                 zeros(n_y, n_p)       r_A_x               r_A_y          # y
        ]
    

    Y = [ -G_mu                 zeros(N_μ, n_p)        -G_x               zeros(N_μ, n_y)         ; # μ
        zeros(n_p, N_μ)           zeros(n_p, n_p)                   zeros(n_p, N_x)     -R_p_y        # p;
        zeros(N_x, N_μ)        Ft_p                  Ft_xx          zeros(N_x, n_y)         ; # x
        r_A_mu                 zeros(n_y, n_p)       r_A_x               r_A_y          # y
      ]

    return u, J, Y

end

function Residual(dmodel, v::AbstractVector; diff=false)
    
    n1 = N_μ = length(dmodel.G.grid)
    n2 = n_p = length(dmodel.hmodel.factories[:projection].equations)
    n3 = N_x = length(dmodel.F.grid)
    n4 = n_y = length(dmodel.hmodel.symbols[:aggregate])
    
    n_z = length(dmodel.hmodel.symbols[:exogenous])
    n_x = length(dmodel.F.x0.data[1])

    μ = v[1:N_μ]
    p = SVector(v[n1+1:n1+n2]...)
    data = copy( reinterpret(SVector{n_x, Float64}, v[n1+n2+1:n1+n2+n3]))
    x = MSM(data, dmodel.F.x0.sizes)
    y = SVector(v[n1+n2+n3+1:end]...)
    

    u = Dolark.Unknown(μ, p, x, y)

    res, J = Residual(dmodel, u)
    r = flatten(res)

    if diff
        return r, J
    else
        return r
    end

end

import Dolo:MSM
import Base: *


function flatten(u::Dolark.Unknown)
    cat(
        u.μ[:],
        u.p,
        cat(u.x.data...; dims=1),
        u.y
    ;dims=1
    )
end

# function unflatten(u::Dolark.SJJac, v::AbstractVector)

#     n1 = length(u.E.μ)
#     μ = reshape(v[1:n1], size(u.E.μ)...)
#     n2 = length(u.F_p.data[1])
#     p = SVector(v[n1+1:n1+n2]...)
#     n_x = size(u.F_p.data[1],1)
#     n3 = length(u.F_p.data)*n_x
#     data = copy( reinterpret(SVector{n_x, Float64}, v[n1+n2+1:n1+n2+n3]))
#     x = MSM(data, u.F_p.sizes)
#     y = SVector(v[n1+n2+n3+1:end]...)
    
#     u = Dolark.Unknown(μ, p, x, y)

#     return u

# end



using IterativeSolvers





using Plots


import Base: +

function +(a::Dolark.Unknown, b::Dolark.Unknown)
    p = a.p + b.p
    x = a.x + b.x
    y = a.y + b.y
    μ = a.μ + b.μ
    return Dolark.Unknown(μ,p,x,y)
end

using FiniteDiff
function proto_solve_steady_state(dmodel, u0)

    backsteps = [2.0^(-i) for i=0:10]

    u0_ = flatten(u0)

    for i=1:10

        println(u0_[end])
        r_, J = Residual(dmodel, u0_; diff=true)

        ε = maximum(abs, r_)

        println("r: $(ε)")
        M = convert(Matrix, J)

        # r_ = Residual(dmodel, u0_)
        # M = FiniteDiff.finite_difference_jacobian(o->Residual(dmodel, o), u0_)
        # ε = maximum(abs, r_)
        # println("r: $(ε)")
        
        # δ_ = -M \ r_
        δ_ = -gmres(J,r_)

        for λ in backsteps
            guess = u0_ + δ_*λ  # (J,λ*δ_)
            εg_ = [1.0]
            try
                rg = Residual(dmodel, guess)
                εg_[1] = maximum(abs,rg)
            catch 
                εg_[1] = 1000.0
            end
            εg = εg_[1]
            if εg<ε
                println(εg)
                u0_ = guess
                break
            end
        end
        # δ = unflatten(J, δ_)

        # u0 = u0 + δ

    end

end


