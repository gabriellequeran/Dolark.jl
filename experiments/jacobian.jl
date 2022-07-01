import Dolark

hmodel = Dolark.HModel("models/ayiagari.yaml")

import Dolo
using FiniteDiff
using StaticArrays


function E_(i,n)
    z = zeros(n)
    z[i] = 1
    return z
end


function LinearThing_power(lt::Dolo.LinearThing, pwr::Int64, x::Vector{Float64})
    for k in 1:pwr
        x = lt*x
    end
    return x
end


function LinearMaps_power(lm,pwr::Int64,μ)
    for i in 1:pwr
        μ = lm*μ
    end
    return μ
end


function fill_a_matrix_by_blocks!(the_matrix, line, column, block_height, block_wide, matrix_to_add)
    the_matrix[(1+(line-1)*block_height):(line*block_height),(1+(column-1)*block_wide):column*block_wide] += matrix_to_add
end


function eval_block(the_matrix, line, column, block_height, block_wide)
    return the_matrix[(1+(line-1)*block_height):(line*block_height),(1+(column-1)*block_wide):column*block_wide]
end


function fill_second_line_of_dM!(dM; dX=dX, ∂G_∂x = ∂G_∂x, T=T)
    for j in 2:(T+1)
        new_matrix = reduce(hcat,[∂G_∂x*dX[j-1][:,k] for k in 1:n_y])
        fill_a_matrix_by_blocks!(dM, 2, j, n_x, n_y, new_matrix)
    end
end


function fill_strictly_under_the_diagonal_of_dM!(dM; dX= dX, ∂G_∂μ= ∂G_∂μ, ∂G_∂x=∂G_∂x, T=T)
    for j in 2:T
        above = dM[(1+(j-1)*n_x):(j*n_x),(1+(j-1)*n_y):j*n_y]
        for i in j+1:(T+1)
            above = reduce(hcat,[∂G_∂μ * above[:,k] for k in 1:n_y])
            fill_a_matrix_by_blocks!(dM,i,j,n_x,n_y, above)
        end
    end
end


function fill_the_rest_above_the_diagonal_of_dM!(dM; ∂G_∂μ= ∂G_∂μ, T=T)
    for i in 3:(T+1)
        for j in 3:(T+1)
            block_of_second_line = eval_block(dM, 2, j, n_x, n_y)
            next_matrix = reduce(hcat,[LinearMaps_power(∂G_∂μ, i-2 , block_of_second_line[:,k]) for k in 1:n_y]) + eval_block(dM, i-1, j-1, n_x, n_y)
            fill_a_matrix_by_blocks!(dM, i, j, n_x, n_y, next_matrix)
        end
    end
end


function fill_dM!(dM; dX= dX, ∂G_∂μ= ∂G_∂μ, ∂G_∂x=∂G_∂x, T=T)
    fill_second_line_of_dM!(dM; dX=dX, ∂G_∂x = ∂G_∂x, T=T)
    fill_the_rest_above_the_diagonal_of_dM!(dM; ∂G_∂μ= ∂G_∂μ, T=T)
    fill_strictly_under_the_diagonal_of_dM!(dM; dX= dX, ∂G_∂μ= ∂G_∂μ, ∂G_∂x=∂G_∂x, T=T)
end


# filling the jacobian matrix
function fill_∂H_∂Y!(∂H_∂Y; dM=dM, dX=dX, dm=dm)
    for i in 1:(T+1)
        for j in 1:(T+1)
            if i==j
                new_matrix = reduce(hcat,[∂A_∂y*E_(k, n_y) + ∂A_∂x * (F_p1 * (r_p_y*E_(k, n_y))) + [∂A_∂μ*(eval_block(dM, i, j, n_x, n_y)[:,k])] for k in 1:n_y])
                fill_a_matrix_by_blocks!(∂H_∂Y, i, j, n_y, n_y, new_matrix)
            elseif j>i
                new_matrix = reduce(hcat,[∂A_∂x * dX[j-i][:,k] + [∂A_∂μ*(eval_block(dM, i, j, n_x, n_y)[:,k])] for k in 1:n_y])
                fill_a_matrix_by_blocks!(∂H_∂Y, i, j, n_y, n_y, new_matrix)
            else
                new_matrix = reduce(hcat,[[∂A_∂μ*(eval_block(dM, i, j, n_x, n_y)[:,k] + dm[i-j][:,k]) ] for k in 1:n_y])
                fill_a_matrix_by_blocks!(∂H_∂Y, i, j, n_y, n_y, new_matrix)
            end
        end
    end
end


#inputs
T=150
yss = SVector(52.693273233617525)
z = SVector(0.)

function compute_jacobian_wrt_Y(hmodel, yss, z; T=150)

    #parms of interest
    parm = hmodel.calibration[:parameters]
    p, r_p_y = Dolark.projection(hmodel, Val{(0,1)}, yss, z, parm)
    r,w = p
    p = SVector{length(p),Float64}(p...)
    Dolo.set_calibration!(hmodel.agent; r=r, w=w)
    sol_agent = Dolo.improved_time_iteration(hmodel.agent; verbose=false)
    dmodel = Dolark.discretize(hmodel, sol_agent) 
    μss = μ = Dolo.ergodic_distribution(hmodel.agent, sol_agent)
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

    #creating a vector dX of matrices homogeneous with some ∂x_∂y that helps to compute A_x dx and A_μ dμ
    dX = Vector{Matrix{Float64}}()
    dX_i = reduce(hcat,(-[L*((J\F_p1)*(r_p_y*E_(k, n_y))) - (J\F_p2)*(r_p_y*E_(k, n_y)) for k in 1:n_y]))
    push!(dX,dX_i)
    for t=1:(T-1)
        dX_i = reduce(hcat,([L*dX_i[:,k] for k in 1:n_y]))
        push!(dX,dX_i)
    end

    #creating a vector dm which contains matrices (for different t) extracted from the total ∂μ_∂y 
    dm = Vector{Matrix{Float64}}()
    dm_i = reduce(hcat,([∂G_∂x*(F_p1*(r_p_y*E_(k, n_y))) for k in 1:n_y]))
    push!(dm,dm_i)
    for t=1:(T-1)
        dm_i = reduce(hcat,([∂G_∂μ*dm_i[:,k] for k in 1:n_y]))
        push!(dm,dm_i)
    end

    #creating a matrix containing the rest of ∂μ_∂y
    dM = zeros((T+1) * n_x, (T+1) * n_y)
    fill_dM!(dM)

    #computing the jacobian
    ∂H_∂Y = zeros((T+1) * n_y, (T+1) * n_y)
    fill_∂H_∂Y!(∂H_∂Y; dM=dM, dX=dX, dm=dm)

    return ∂H_∂Y
end


@time compute_jacobian_wrt_Y(hmodel, yss, z; T=150)


