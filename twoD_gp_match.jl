import DifferentialEquations, PyCall, PyPlot, Distributions, Optim, Combinatorics

PyCall.@pyimport GPy.kern as gkern
PyCall.@pyimport GPy.models as gmodels


mutable struct GPparset
	id::Int
	speciesnum::Int
	intercount::Int
	parents::Array{Int, 1}
	lik::Float64
	modaic::Float64
	modbic::Float64
	aicweight::Float64
	bicweight::Float64
end


function simulate()
	function odesys(t,u,du) # Define ODE system
		du[1] = 0.1-0.4*u[1]+2.0*((u[5]^2.0)/(1.5^2.0+u[5]^2.0))
		du[2] = 0.2-0.4*u[2]+1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[5]/2.0)^1.0))
		du[3] = 0.2-0.4*u[3]+2.0*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))
		du[4] = 0.4-0.1*u[4]+1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[3]/1.0)^2.0))
		du[5] = 0.3-0.3*u[5]+2.0*((u[4]^2.0)/(1.0^2.0+u[4]^2.0))*(1.0/(1.0+(u[2]/0.5)^3.0))
	end

	u0 = [1.0;0.5;1.0;1.5;0.5] # Define initial conditions
	tspan = (0.0,20.0) # Define timespan for solving ODEs
	prob = DifferentialEquations.ODEProblem(odesys,u0,tspan) # Formalise ODE problem

	sol = DifferentialEquations.solve(prob, DifferentialEquations.RK4(), dt=1.0) # Solve ODEs with RK4 solver
	x = reshape(sol.t,(length(sol.t),1))
	y = hcat(sol.u...)'
	y .+= reshape(rand(Distributions.Normal(0, 0.05), length(y)), size(y))
	x, y
end


function interpolate(x,y)
	n = size(y,2)

	kernel = gkern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
	xmu = Array{Float64}(size(y))
	xvar = Array{Float64}(size(y))
	xdotmu = Array{Float64}(size(y))
	xdotvar = Array{Float64}(size(y))

	for i = 1:n
		Y = reshape(y[:,i],(length(x),1))

		m = gmodels.GPRegression(x,Y,kernel)
		m[:optimize_restarts](num_restarts = 5, verbose=false)
		m[:plot](plot_density=false)

		vals = m[:predict](x)
		deriv = m[:predict_jacobian](x)

		xmu[:,i] = reshape(vals[1], size(y,1))
		xvar[:,i] = reshape(vals[2], size(y,1))
		xdotmu[:,i] = reshape(deriv[1], size(y,1))
		xdotvar[:,i] = reshape(deriv[2], size(y,1))
	end

	xmu, xvar, xdotmu, xdotvar
end


function construct_gpparsets(numspecies, maxinter; selfinter=false)
	allparents = collect(1:numspecies)
	gpparsets::Array{GPparset,1} = []
	count::Int = 0

	if selfinter
		for i = 1:numspecies
			for k = 1:maxinter
				for l in Combinatorics.combinations(allparents, k)
					count += 1
					push!(gpparsets, GPparset(count, i, k, l, 0.0, 0.0, 0.0, 0.0, 0.0))
				end
			end
		end

	else
		for i = 1:numspecies
			for k = 1:maxinter
				for l in Combinatorics.combinations(filter(e -> e ≠ i, allparents), k)
					count += 1
					push!(gpparsets, GPparset(count, i, k, l, 0.0, 0.0, 0.0, 0.0, 0.0))
				end
			end
		end
	end

	reshape(gpparsets,(:,numspecies))
end


function compute_lik!(gppar::GPparset,x,y)
	kernel = gkern.RBF(input_dim=gppar.intercount, variance=1, lengthscale=1)
	m = gmodels.GPRegression(x,y,kernel)
	m[:optimize_restarts](num_restarts = 5, verbose=false)
	# m[:plot](plot_density=false)
	gppar.lik = m[:log_likelihood]()
	gppar.modaic =  2 * gppar.intercount - 2 * gppar.lik
	gppar.modbic = log(size(y)[1]) * gppar.intercount - 2 * gppar.lik
end


function get_all_lik!(gpparsets,y,xdotmu)
	@progress "Optimising GPs" for gppar in gpparsets
		X = y[:,gppar.parents]
		Y = reshape(xdotmu[:,gppar.speciesnum] - fixparm[1,gppar.speciesnum] + fixparm[2,gppar.speciesnum] .* y[:,gppar.speciesnum],(:,1))
		compute_lik!(gppar,X,Y)
	end
end


function weight_models!(gpparsets)
	for i = 1:size(gpparsets,2)
		minaic = minimum(p.modaic for p in gpparsets[:,i])
		minbic = minimum(p.modbic for p in gpparsets[:,i])
		denomsumaic = sum(exp((-p.modaic + minaic) / 2) for p in gpparsets[:,i])
		denomsumbic = sum(exp((-p.modbic + minbic) / 2) for p in gpparsets[:,i])

		for j in gpparsets[:,i]
			j.aicweight = exp((-j.modaic + minaic) / 2) / denomsumaic
			j.bicweight = exp((-j.modbic + minbic) / 2) / denomsumbic
		end
	end
end


"""
Create edgeweight array with columns:\\
Target, Source, Weight
"""
function weight_edges(gpparsets)
	numsp = size(gpparsets,2)
	edgeweights = hcat(repeat(collect(1:numsp),inner=numsp),
						repeat(collect(1:numsp),outer=numsp),
						zeros(numsp^2),
						zeros(numsp^2))

	for gpparset in gpparsets
		for (i, parent) in enumerate(gpparset.parents)
			row = (gpparset.speciesnum - 1) * numsp + (parent-1) + 1
			edgeweights[row,3] += gpparset.aicweight
			edgeweights[row,4] += gpparset.bicweight
		end
	end
	edgeweights
end


"""
    get_true_ranks(trueparents, parsets)
Create ranks array with columns representing each specie and rows:\\
Specie Number, ID, Log_likelihood, AIC, BIC, Likrank, AICrank, BICrank, AICweight, BICweight
"""
function get_true_ranks(trueparents, parsets)
	ranks = []
	for tp in trueparents
		for parset in parsets[:, tp.speciesnum]
			if tp.parents == parset.parents
				likrank = find(reverse(sort([i.lik for i in parsets[:, tp.speciesnum]])) .== parset.lik)
				aicrank = find(sort([i.modaic for i in parsets[:, tp.speciesnum]]) .== parset.modaic)
				bicrank = find(sort([i.modbic for i in parsets[:, tp.speciesnum]]) .== parset.modbic)
				if tp.speciesnum == 1
					ranks = [parset.speciesnum, parset.id, parset.lik, parset.modaic, parset.modbic,
								likrank, aicrank, bicrank, parset.aicweight, parset.bicweight]
				else
					ranks = hcat(ranks, [parset.speciesnum, parset.id, parset.lik, parset.modaic, parset.modbic,
											likrank, aicrank, bicrank, parset.aicweight, parset.bicweight])
				end
				break
			end
		end
	end
	ranks
end


"""
    get_best_id(gpparsets)
###Create array with top scoring models. Columns representing each specie and rows:\\
Specie Number, ID of max log_lik mod, ID of min aic mod, ID of min bic mod,
Log_lik of max log_lik mod, AIC of min aic mod, BIC of min bic mod
"""
function get_best_id(gpparsets)
	bestlist = []
	for j = 1:size(gpparsets,2)
		row = find([i.lik for i in gpparsets[:,j]] .== maximum(i.lik for i in gpparsets[:,j]))[1]
		distid = gpparsets[row].id
		row = find([i.modaic for i in gpparsets[:,j]] .== minimum(i.modaic for i in gpparsets[:,j]))[1]
		aicid = gpparsets[row].id
		row = find([i.modbic for i in gpparsets[:,j]] .== minimum(i.modbic for i in gpparsets[:,j]))[1]
		bicid = gpparsets[row].id
		if j == 1
			bestlist = [j, distid, aicid, bicid, gpparsets[distid].lik, gpparsets[aicid].modaic, gpparsets[bicid].modbic]
		else
			bestlist = hcat(bestlist,[j, distid, aicid, bicid, gpparsets[distid].lik, gpparsets[aicid].modaic, gpparsets[bicid].modbic])
		end
	end
	bestlist
end


################################################################################


x, y = simulate()

const numspecies = size(y,2)
const maxinter = 2
const interactions = [:Activation, :Repression]
const fixparm = [0.1 0.2 0.2 0.4 0.3; 0.4 0.4 0.4 0.1 0.3]
const trueparents = [GPparset(0, 1, 1, [5], 0.0, 0.0, 0.0, 0.0, 0.0),
						GPparset(0, 2, 2, [1, 5], 0.0, 0.0, 0.0, 0.0, 0.0),
						GPparset(0, 3, 1, [1], 0.0, 0.0, 0.0, 0.0, 0.0),
						GPparset(0, 4, 2, [1, 3], 0.0, 0.0, 0.0, 0.0, 0.0),
						GPparset(0, 5, 2, [2, 4], 0.0, 0.0, 0.0, 0.0, 0.0)]

xmu, xvar, xdotmu, xdotvar = interpolate(x, y)

gpparsets = construct_gpparsets(numspecies, maxinter; selfinter=false)

get_all_lik!(gpparsets,y,xdotmu)

weight_models!(gpparsets)

edgeweights = weight_edges(gpparsets)

ranks = get_true_ranks(trueparents, gpparsets)

bestmodels = get_best_id(gpparsets)



# import Plots
# Plots.scatter(x[:,1],xdotmu[:,1])
