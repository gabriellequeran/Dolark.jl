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

function create_dm(n, r_p, ∂G_∂x, ∂G_∂μ, F_p1; T=T)
    dm = Vector{Matrix{Float64}}()
    dm_i = reduce(hcat,([∂G_∂x*(F_p1*(r_p*E_(k, n))) for k in 1:n]))
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
    for i in 3:(T+1)
        for j in 3:(T+1)
            block_of_second_line = eval_block(dM, 2, j, n_x, n)
            next_matrix = reduce(hcat,[LinearMaps_power(∂G_∂μ, i-2 , block_of_second_line[:,k]) for k in 1:n]) + eval_block(dM, i-1, j-1, n_x, n)
            fill_a_matrix_by_blocks!(dM, i, j, n_x, n, next_matrix)
        end
    end
end


function fill_dM!(dM, n, dX; ∂G_∂μ= ∂G_∂μ, ∂G_∂x=∂G_∂x, T=T, n_x=n_x)
    t0 = time()
    fill_second_line_of_dM!(dM, n, dX; ∂G_∂x = ∂G_∂x, T=T, n_x=n_x)
    t1 = time()
    println("time to fill second line of dM: ", t1-t0)
    fill_the_rest_above_the_diagonal_of_dM!(dM, n; ∂G_∂μ= ∂G_∂μ, T=T, n_x=n_x)
    t2 = time()
    println("time to fill above the diagonal of dM: ", t2-t1)
    fill_strictly_under_the_diagonal_of_dM!(dM, n; ∂G_∂μ= ∂G_∂μ, ∂G_∂x=∂G_∂x, T=T, n_x=n_x)
    t3 = time()
    println("time to fill under the diagonal of dM: ", t3-t2)
end


# filling the jacobian matrices
function fill_∂H_∂YorZ!(∂H_∂YorZ, n, dX, ∂A_∂yorz, r_p, dM, dm; T=T, F_p1=F_p1, ∂A_∂x= ∂A_∂x, ∂A_∂μ= ∂A_∂μ, n_x=n_x, n_y=n_y)
    for i in 1:(T+1)
        for j in 1:(T+1)
            if i==j
                new_matrix = reduce(hcat,[∂A_∂yorz*E_(k, n) + ∂A_∂x * (F_p1 * (r_p*E_(k, n))) + [∂A_∂μ*(eval_block(dM, i, j, n_x, n)[:,k])] for k in 1:n])
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

function compute_jacobians(hmodel, yss, z; T=300)

    #parms of interest
    t0 = time()
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
    t1 = time()
    println("time to find the parms of interest: ", t1-t0)

    #creating vectors dX_Y (or dX_Z) of matrices homogeneous with some ∂x_∂y (or ∂x_∂z) that help to compute A_x dx and A_μ dμ
    dX_Y = create_dX(n_y, r_p_y,  J, L, F_p1, F_p2; T=T)
    dX_Z = create_dX(n_z, r_p_z,  J, L, F_p1, F_p2; T=T)
    t2 = time()
    println("time to find dX: ", t2-t1)

    #creating vectors dm_Y (or dm_Z) which contain matrices (for different t) extracted from the total ∂μ_∂y (or ∂μ_∂z)
    dm_Y = create_dm(n_y, r_p_y, ∂G_∂x, ∂G_∂μ, F_p1; T=T)
    dm_Z = create_dm(n_z, r_p_z, ∂G_∂x, ∂G_∂μ, F_p1; T=T)
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
    fill_∂H_∂YorZ!(∂H_∂Y, n_y, dX_Y, ∂A_∂y, r_p_y, dM_Y, dm_Y; T=T, F_p1=F_p1, ∂A_∂x= ∂A_∂x, ∂A_∂μ= ∂A_∂μ, n_x=n_x, n_y=n_y)
    ∂H_∂Z = zeros((T+1) * n_y, (T+1) * n_z)
    fill_∂H_∂YorZ!(∂H_∂Z, n_z, dX_Z, ∂A_∂z, r_p_z, dM_Z, dm_Z;T=T, F_p1=F_p1, ∂A_∂x= ∂A_∂x, ∂A_∂μ= ∂A_∂μ, n_x=n_x, n_y=n_y)
    t5 = time()
    println("time to fill ∂H_∂?: ", t5-t4)

    return ∂H_∂Y, ∂H_∂Z
end


∂H_∂Y, ∂H_∂Z = compute_jacobians(hmodel, yss, z; T=299) # around 3 minutes. The calculation of dM is by far the most costful. 

∂H_∂Y

∂H_∂Z

# Practical case of impulse responses :
using Plots
dZ = [0. for k in 1:300]
dZ[1] = 1e-5

dY = -∂H_∂Y \ ∂H_∂Z * dZ
plot(dY)







# #parms of interest
# parm = hmodel.calibration[:parameters]
# p, r_p_y, r_p_z = Dolark.projection(hmodel, Val{(0,1,2)}, yss, z, parm)
# r,w = p
# p = SVector{length(p),Float64}(p...)
# Dolo.set_calibration!(hmodel.agent; r=r, w=w)
# sol_agent = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
# dmodel = Dolark.discretize(hmodel, sol_agent) 
# μss = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
# xss = Dolo.MSM([sol_agent.dr(i, dmodel.F.s0) for i=1:length(dmodel.F.grid.exo)])
# J = Dolo.df_A(dmodel.F, xss, xss; exo=(p,p))
# L = Dolo.df_B(dmodel.F, xss, xss; exo=(p,p))
# F_p1, F_p2 = Dolo.df_e(dmodel.F, xss, xss, p, p)
# μ, ∂G_∂μ, ∂G_∂x = dmodel.G(μss,xss; diff=true)
# A, ∂A_∂μ, ∂A_∂x, ∂A_∂y, ∂A_∂z = Dolark.𝒜(dmodel, μss, xss, yss, z; diff=true)
# Dolo.mult!(L,-1)
# Dolo.prediv!(L,J)
# n_y = length(yss)
# n_x = length(xss.data)
# n_z = length(z)

# function compute_jacobians(J, L, F_p1, F_p2, n_x, n_y, n_z, r_p_y, r_p_z, ∂A_∂x, ∂A_∂y, ∂A_∂z, ∂A_∂μ, ∂G_∂x, ∂G_∂μ; T=300)

#     #creating vectors dX_Y (or dX_Z) of matrices homogeneous with some ∂x_∂y (or ∂x_∂z) that help to compute A_x dx and A_μ dμ
#     dX_Y = create_dX(n_y, r_p_y,  J, L, F_p1, F_p2; T=T)
#     dX_Z = create_dX(n_z, r_p_z,  J, L, F_p1, F_p2; T=T)

#     #creating vectors dm_Y (or dm_Z) which contain matrices (for different t) extracted from the total ∂μ_∂y (or ∂μ_∂z)
#     dm_Y = create_dm(n_y, r_p_y, ∂G_∂x, ∂G_∂μ, F_p1; T=T)
#     dm_Z = create_dm(n_z, r_p_z, ∂G_∂x, ∂G_∂μ, F_p1; T=T)

#     #creating matrices containing the rest of ∂μ_∂y or ∂μ_∂z
#     dM_Y = zeros((T+1) * n_x, (T+1) * n_y)
#     fill_dM!(dM_Y, n_y, dX_Y; ∂G_∂μ= ∂G_∂μ, ∂G_∂x=∂G_∂x, T=T, n_x=n_x)

#     dM_Z = zeros((T+1) * n_x, (T+1) * n_z)
#     fill_dM!(dM_Z, n_z, dX_Z; ∂G_∂μ= ∂G_∂μ, ∂G_∂x=∂G_∂x, T=T, n_x=n_x)

#     #computing the jacobians
#     ∂H_∂Y = zeros((T+1) * n_y, (T+1) * n_y)
#     fill_∂H_∂YorZ!(∂H_∂Y, n_y, dX_Y, ∂A_∂y, r_p_y, dM_Y, dm_Y; T=T, F_p1=F_p1, ∂A_∂x= ∂A_∂x, ∂A_∂μ= ∂A_∂μ, n_x=n_x, n_y=n_y)

#     ∂H_∂Z = zeros((T+1) * n_y, (T+1) * n_z)
#     fill_∂H_∂YorZ!(∂H_∂Z, n_z, dX_Z, ∂A_∂z, r_p_z, dM_Z, dm_Z;T=T, F_p1=F_p1, ∂A_∂x= ∂A_∂x, ∂A_∂μ= ∂A_∂μ, n_x=n_x, n_y=n_y)

#     return ∂H_∂Y, ∂H_∂Z
# end


