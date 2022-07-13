import Dolark

hmodel = Dolark.HModel("models/ayiagari.yaml")

import Dolo
using FiniteDiff
using StaticArrays

"""
Creates a vector of length n with 0 everywhere except a 1 in position i
# Arguments
* `n`: output length
* `i`: position of the one in the vector
"""
function E_(i,n)
    z = zeros(n)
    z[i] = 1
    return z
end

"""
Computes the product of an object LinearThing, lt, at a certain power (pwr), with a vector x
"""
function LinearThing_power(lt::Dolo.LinearThing, pwr::Int64, x::Vector{Float64})
    for k in 1:pwr
        x = lt*x
    end
    return x
end

"""
Computes the product of a FunctionMap, lm, at a certain power (pwr), with a vector μ
# Arguments
* `lm::LinearMaps.FunctionMap{Float64}`: function map 
* `pwr::Int64`: power
* `μ::Vector{Float64}`: vector to be multiplied by the function map at a certain power
"""
function LinearMaps_power(lm,pwr::Int64,μ)
    for i in 1:pwr
        μ = lm*μ
    end
    return μ
end

"""
Fills in place a matrix by blocks, in adding to one of its blocks a certain matrix
# Arguments
* `the_matrix::Matrix{Float64}`: the matrix in which the completion will be done 
* `line::Int64`: the line of the block (from the_matrix) to be completed (ie. if vertically we have two blocks of 3 lines, the second block will be indexed by line=2)
* `column::Int64`: the column of the block (from the_matrix) to be completed
* `block_height::Int64`: the height of the blocks in the_matrix
* `block_width::Int64`: the width of the blocks in the_matrix
* `matrix_to_add::Matrix{Float64}`: matrix to be added to a block of the_matrix
"""
function fill_a_matrix_by_blocks!(the_matrix, line, column, block_height, block_width, matrix_to_add)
    the_matrix[(1+(line-1)*block_height):(line*block_height),(1+(column-1)*block_width):column*block_width] += matrix_to_add
end

"""
Finds a block in a matrix
# Arguments
* `the_matrix::Matrix{Float64}`: the matrix in which the block will be found 
* `line::Int64`: the line of the block (from the_matrix) to be extracted
* `column::Int64`: the column of the block (from the_matrix) to be extracted
* `block_height::Int64`: the height of the blocks in the_matrix
* `block_width::Int64`: the width of the blocks in the_matrix
# returns
* the block located at the intersection of line and column
"""
function eval_block(the_matrix, line, column, block_height, block_width)
    return the_matrix[(1+(line-1)*block_height):(line*block_height),(1+(column-1)*block_width):column*block_width]
end

#The description of the 8 following functions can be found in impulse_responses.ipynb.

function create_dX(n,r_p, J, L, F_p1, F_p2; T=T)
    dX = Vector{Matrix{Float64}}()
    dX_i = reduce(hcat,(-[L*((J\F_p1)*(r_p*E_(k, n))) + (J\F_p2)*(r_p*E_(k, n)) for k in 1:n]))
    push!(dX,dX_i)
    for t=1:(T-1)
        dX_i = reduce(hcat,([L*dX_i[:,k] for k in 1:n]))
        push!(dX,dX_i)
    end
    return dX
end

function create_dm(n, r_p, ∂G_∂x, ∂G_∂μ, F_p1, J; T=T)
    dm = Vector{Matrix{Float64}}()
    dm_i = reduce(hcat,(-[∂G_∂x*(J\F_p1*(r_p*E_(k, n))) for k in 1:n]))
    push!(dm,dm_i)
    for t=1:(T-1)
        dm_i = reduce(hcat,([∂G_∂μ*dm_i[:,k] for k in 1:n]))
        push!(dm,dm_i)
    end
    return dm
end

function fill_second_line_of_dM!(dM, n, dX; ∂G_∂x = ∂G_∂x, T=T, n_x=n_x)
    for j in 2:(T+1)
        new_matrix = reduce(hcat,[∂G_∂x*dX[j-1][:,k] for k in 1:n])
        fill_a_matrix_by_blocks!(dM, 2, j, n_x, n, new_matrix)
    end
end


function fill_strictly_under_the_diagonal_of_dM!(dM, n; ∂G_∂μ= ∂G_∂μ, ∂G_∂x=∂G_∂x, T=T, n_x=n_x)
    for j in 2:T
        above = dM[(1+(j-1)*n_x):(j*n_x),(1+(j-1)*n):j*n]
        for i in j+1:(T+1)
            above = reduce(hcat,[∂G_∂μ * above[:,k] for k in 1:n])
            fill_a_matrix_by_blocks!(dM,i,j,n_x,n, above)
        end
    end
end


function fill_the_rest_above_the_diagonal_of_dM!(dM, n; ∂G_∂μ= ∂G_∂μ, T=T, n_x=n_x)
    for j in 3:(T+1)
        power_of_the_block_of_second_line = eval_block(dM, 2, j, n_x, n)
        for i in 3:(T+1)
            power_of_the_block_of_second_line = reduce(hcat,[∂G_∂μ * power_of_the_block_of_second_line[:,k] for k in 1:n])
            next_matrix = power_of_the_block_of_second_line + eval_block(dM, i-1, j-1, n_x, n)
            fill_a_matrix_by_blocks!(dM, i, j, n_x, n, next_matrix)
        end
    end
end

function fill_dM!(dM, n, dX; ∂G_∂μ= ∂G_∂μ, ∂G_∂x=∂G_∂x, T=T, n_x=n_x)
    fill_second_line_of_dM!(dM, n, dX; ∂G_∂x = ∂G_∂x, T=T, n_x=n_x)
    fill_the_rest_above_the_diagonal_of_dM!(dM, n; ∂G_∂μ= ∂G_∂μ, T=T, n_x=n_x)
    fill_strictly_under_the_diagonal_of_dM!(dM, n; ∂G_∂μ= ∂G_∂μ, ∂G_∂x=∂G_∂x, T=T, n_x=n_x)
end


# filling the jacobian matrices
function fill_∂H_∂YorZ!(∂H_∂YorZ, n, dX, ∂A_∂yorz, r_p, dM, dm; T=T, F_p1=F_p1, ∂A_∂x= ∂A_∂x, ∂A_∂μ= ∂A_∂μ, n_x=n_x, n_y=n_y, J=J)
    for i in 1:(T+1)
        for j in 1:(T+1)
            if i==j
                new_matrix = reduce(hcat,[∂A_∂yorz*E_(k, n) - ∂A_∂x * (J\F_p1 * (r_p*E_(k, n))) + [∂A_∂μ*(eval_block(dM, i, j, n_x, n)[:,k])] for k in 1:n])
                fill_a_matrix_by_blocks!(∂H_∂YorZ, i, j, n_y, n, new_matrix)
            elseif j>i
                new_matrix = reduce(hcat,[∂A_∂x * dX[j-i][:,k] + [∂A_∂μ*(eval_block(dM, i, j, n_x, n)[:,k])] for k in 1:n])
                fill_a_matrix_by_blocks!(∂H_∂YorZ, i, j, n_y, n, new_matrix)
            else
                new_matrix = reduce(hcat,[[∂A_∂μ*(eval_block(dM, i, j, n_x, n)[:,k] + dm[i-j][:,k]) ] for k in 1:n])
                fill_a_matrix_by_blocks!(∂H_∂YorZ, i, j, n_y, n, new_matrix)
            end
        end
    end
end


#inputs
yss = SVector(52.693273233617525)
z = SVector(0.)

#parms of interest
parm = hmodel.calibration[:parameters]
p, r_p_y, r_p_z = Dolark.projection(hmodel, Val{(0,1,2)}, yss, z, parm)
r,w = p
p = SVector{length(p),Float64}(p...)
Dolo.set_calibration!(hmodel.agent; r=r, w=w)
sol_agent = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
dmodel = Dolark.discretize(hmodel, sol_agent) 
μss = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
xss = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])
J = Dolo.df_A(dmodel.F, xss, xss; exo=(p,p))
L = Dolo.df_B(dmodel.F, xss, xss; exo=(p,p))
F_p1, F_p2 = Dolo.df_e(dmodel.F, xss, xss, p, p)
μ, ∂G_∂μ, ∂G_∂x = dmodel.G(μss,xss; diff=true)
A, ∂A_∂μ, ∂A_∂x, ∂A_∂y, ∂A_∂z = Dolark.𝒜(dmodel, μss, xss, yss, z; diff=true)
Dolo.mult!(L,-1)
Dolo.prediv!(L,J)
n_y = length(yss)
n_x = length(xss.data)
n_z = length(z)
T=299

function compute_jacobians(n_x, n_y, n_z, r_p_y, r_p_z, J, L, F_p1, F_p2, ∂G_∂x, ∂G_∂μ, ∂A_∂x, ∂A_∂y, ∂A_∂z, ∂A_∂μ; T=T)

    #creating vectors dX_Y (or dX_Z) of matrices homogeneous with some ∂x_∂y (or ∂x_∂z) that help to compute A_x dx and A_μ dμ
    t1 = time()
    dX_Y = create_dX(n_y, r_p_y,  J, L, F_p1, F_p2; T=T)
    dX_Z = create_dX(n_z, r_p_z,  J, L, F_p1, F_p2; T=T)
    t2 = time()
    println("time to find dX: ", t2-t1)

    #creating vectors dm_Y (or dm_Z) which contain matrices (for different t) extracted from the total ∂μ_∂y (or ∂μ_∂z)
    dm_Y = create_dm(n_y, r_p_y, ∂G_∂x, ∂G_∂μ, F_p1, J; T=T)
    dm_Z = create_dm(n_z, r_p_z, ∂G_∂x, ∂G_∂μ, F_p1, J; T=T)
    t3 = time()
    println("time to find dm: ", t3-t2)

    #creating matrices containing the rest of ∂μ_∂y or ∂μ_∂z
    dM_Y = zeros((T+1) * n_x, (T+1) * n_y)
    fill_dM!(dM_Y, n_y, dX_Y; ∂G_∂μ= ∂G_∂μ, ∂G_∂x=∂G_∂x, T=T, n_x=n_x)
    dM_Z = zeros((T+1) * n_x, (T+1) * n_z)
    fill_dM!(dM_Z, n_z, dX_Z; ∂G_∂μ= ∂G_∂μ, ∂G_∂x=∂G_∂x, T=T, n_x=n_x)
    t4 = time()
    println("time to find dM: ", t4-t3)

    #computing the jacobians
    ∂H_∂Y = zeros((T+1) * n_y, (T+1) * n_y)
    fill_∂H_∂YorZ!(∂H_∂Y, n_y, dX_Y, ∂A_∂y, r_p_y, dM_Y, dm_Y; T=T, F_p1=F_p1, ∂A_∂x= ∂A_∂x, ∂A_∂μ= ∂A_∂μ, n_x=n_x, n_y=n_y, J=J)
    ∂H_∂Z = zeros((T+1) * n_y, (T+1) * n_z)
    fill_∂H_∂YorZ!(∂H_∂Z, n_z, dX_Z, ∂A_∂z, r_p_z, dM_Z, dm_Z;T=T, F_p1=F_p1, ∂A_∂x= ∂A_∂x, ∂A_∂μ= ∂A_∂μ, n_x=n_x, n_y=n_y, J=J)
    t5 = time()
    println("time to fill ∂H_∂?: ", t5-t4)

    return ∂H_∂Y, ∂H_∂Z
end


∂H_∂Y, ∂H_∂Z = compute_jacobians(n_x, n_y, n_z, r_p_y, r_p_z, J, L, F_p1, F_p2, ∂G_∂x, ∂G_∂μ, ∂A_∂x, ∂A_∂y, ∂A_∂z, ∂A_∂μ; T=299) # around 12s.

∂H_∂Y

∂H_∂Z










# Practical case of impulse responses :
using Plots

## the next two graphs aim at giving an insight about the elements inside the jacobians. 5 columns by jacobian are plotted.
p1 = plot(∂H_∂Y[:,1], title = "Columns of ∂H/∂Y", label = "s=0")
plot!(∂H_∂Y[:,26], label = "s=25")
plot!(∂H_∂Y[:,51], label = "s=50")
plot!(∂H_∂Y[:,76], label = "s=75")
plot!(∂H_∂Y[:,101], label = "s=100")
xlabel!("t")


p2 = plot(∂H_∂Z[:,1], title = "Columns of ∂H/∂Z", label = "s=0")
plot!(∂H_∂Z[:,26], label = "s=25")
plot!(∂H_∂Z[:,51], label = "s=50")
plot!(∂H_∂Z[:,76], label = "s=75")
plot!(∂H_∂Z[:,101], label = "s=100")
xlabel!("t")


## function to initialize dZ with s giving the point where the shock happens
function create_dZ(s; value=1e-5, T=300)
    dZ = [0. for k in 1:T]
    dZ[s] = value
    return dZ
end

## impulse responses of Y for shocks at various times (0, 25, 50, 75 and 100)
plot([-∂H_∂Y \ ∂H_∂Z * create_dZ(k) for k in (1,26,51,76,101)], label=["s=0" "s=25" "s=50" "s=75" "s=100"], ylabel = "dY", xlabel="t", title="Impulse response of Y")

## impulse responses of p for shocks at various times (0, 25, 50, 75 and 100)
dp_ = [dY * r_p_y'+ create_dZ(k) * r_p_z' for k in (1,26,51,76,101)]
dr_, dw_ = [dp_[i][:,1] for i in 1:5], [dp_[i][:,2] for i in 1:5]
plot(dr_, title = "impulse response", ylabel="dr", xlabel="t", label=["s=0" "s=25" "s=50" "s=75" "s=100"])
plot(dw_, title = "impulse response", ylabel="dw", xlabel="t", label=["s=0" "s=25" "s=50" "s=75" "s=100"])


## impulse responses of X for shocks at various times (0, 25, 50, 75 and 100)

using LinearMaps
using Krylov
using LinearAlgebra
import Base.size

function Base.size(lt::Dolo.LinearThing)
    return prod(Dolo.shape(lt))
end

function invert(L, r0; smaxit = 1000, tol_ν = 1e-10, krylov = true)
    if krylov
        u0 = Krylov.gmres(I-LinearMaps.LinearMap(z -> L*z,size(L)[1],size(L)[1]), r0)[1]
    else
        u0 = r0
        for i=1:smaxit
            r0 = L*r0
            u0 += r0 # supposed to be the infinite sum useful to compute an inverse
            if norm(r0)<tol_ν
                break
            end
        end
    end
    return u0
end

F_p = F_p1+F_p2
π_ = [[- J \ F_p * dp_[k][i,:] for i in 1:T+1] for k in 1:5]
dX_ = [[invert(L, π_[k][i])  for i in 1:T+1] for k in 1:5]

## impulse responses of µ for shocks at various times (0, 25, 50, 75 and 100)

dμ_ = [[invert(∂G_∂μ, ∂G_∂x * dX_[k][i]) for i in 1:T+1] for k in 1:5]


## impulse responses of X, once aggregated, for shocks at various times (0, 25, 50, 75 and 100)
DX_ = [[dX_[k][i]' *  dμ_[k][i] for i in 1:T+1] for k in 1:5]
plot(DX_, title = "impulse response", ylabel="dX_agg", xlabel="t", label=["s=0" "s=25" "s=50" "s=75" "s=100"])

