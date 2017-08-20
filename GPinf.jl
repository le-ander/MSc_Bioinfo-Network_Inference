module GPinf

using Juno
import DifferentialEquations, PyCall, PyPlot, Distributions, Optim, Combinatorics

PyCall.@pyimport GPy.kern as gkern
PyCall.@pyimport GPy.models as gmodels
PyCall.@pyimport GPy.util.multioutput as gmulti

export Parset, GPparset,
		simulate, interpolate,
		construct_parsets, construct_ode, construct_ode_osc,
		optimise_params!, optimise_models!, weight_models!,
		weight_edges, get_true_ranks, get_best_id, edgesummary, metricdata


mutable struct Parset
	id::Int
	speciesnum::Int
	intercount::Int
	parents::Array{Int, 1}
	intertype::Array{Symbol,1}
	params::Array{Float64,1}
	dist::Float64
	modaic::Float64
	modbic::Float64
	aicweight::Float64
	bicweight::Float64
end

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


function simulate(odesys, tspan, step; noise=0.0)
	u0 = [1.0;0.5;1.0;0.5;0.5] # Define initial conditions
	prob = DifferentialEquations.ODEProblem(odesys,u0,tspan) # Formalise ODE problem

	sol = DifferentialEquations.solve(prob, DifferentialEquations.RK4(), saveat=step) # Solve ODEs with RK4 solver
	x = reshape(sol.t,(length(sol.t),1))
	y = hcat(sol.u...)'
	if noise ≠ 0.0
		y .+= reshape(rand(Distributions.Normal(0, noise), length(y)), size(y))
	end
	x, y
end


function interpolate(x, y, rmfl::Bool, gpnum::Void)
	kernel = gkern.RBF(input_dim=1)
	xmu = Array{Float64}(size(y))
	xvar = Array{Float64}(size(y))
	xdotmu = Array{Float64}(size(y))
	xdotvar = Array{Float64}(size(y))

	for i = 1:size(y,2)
		Y = reshape(y[:,i],(:,1))

		m = gmodels.GPRegression(x,Y,kernel)

		# m[:rbf]["variance"][:constrain_fixed](1e-1)
		# m[:rbf]["lengthscale"][:constrain_bounded](0.0,5.0)
		# m[:Gaussian_noise]["variance"][:constrain_fixed](0.1)

		m[:optimize_restarts](num_restarts = 5, verbose=false, parallel=false)
		m[:plot](plot_density=false)

		# println(m[:param_array])

		vals = m[:predict](x)
		deriv = m[:predict_jacobian](x)

		xmu[:,i] = vals[1][:]
		xvar[:,i] = vals[2][:]
		xdotmu[:,i] = deriv[1][:]
		xdotvar[:,i] = deriv[2][:]
	end
	if rmfl
		xmu = xmu[2:end-1,:]; xvar = xvar[2:end-1,:]
		xdotmu = xdotmu[2:end-1,:]; xdotvar = xdotvar[2:end-1,:]
		x = x[2:end-1,:]
	end
	x, xmu, xvar, xdotmu, xdotvar
end

function interpolate(x, y, rmfl::Bool, gpnum::Int)
	speciesnum = size(y,2)
	eulersx = Vector{Float64}((length(x))*2)
	eulersy = zeros((length(x))*2,speciesnum)
	Δ = 1e-4
	for (i, val) in enumerate(x)
		eulersx[i*2-1] = val - Δ/2
		eulersx[i*2] = val + Δ/2
	end

	count = zeros(Int, speciesnum)'
	xmu = zeros(size(y))
	xvar = zeros(size(y))
	xdotmu = Array{Float64}(convert(Int,size(eulersy,1)/2),speciesnum)

	for comb in Combinatorics.combinations(1:speciesnum, gpnum)
		ytemp = [reshape(y[:,i],(:,1)) for i in comb]
		icm = gmulti.ICM(input_dim=1,num_outputs=gpnum,kernel=gkern.RBF(1))
		m = gmodels.GPCoregionalizedRegression([x for i in comb],ytemp,kernel=icm)

		# m[:ICM][:rbf]["variance"][:constrain_fixed](1e-1)
		# m[:ICM][:rbf]["lengthscale"][:constrain_bounded](0.0,5.0)
		# m[:mixed_noise][:constrain_fixed](0.1)

		m[:optimize_restarts](num_restarts = 16, verbose=false, parallel=true)

		# println(m[:param_array])

		for (i,species) in enumerate(comb)
			count[species] += 1
			prediction = m[:predict](hcat(x,[i-1 for t in x]), Y_metadata=Dict("output_index" => Int[i-1 for t in x]))
			eulersy[:,species] .+= m[:predict](hcat(eulersx,[i-1 for t in eulersx]), Y_metadata=Dict("output_index" => Int[i-1 for t in eulersx]))[1][:]
			xmu[:,species] .+= prediction[1][:]
			xvar[:,species] .+= prediction[2][:]
		end
	end

	xmu ./= count
	xvar ./= count
	eulersy ./= count
	xnew = x[2:end-1,:]

	for i = 1:length(xdotmu)
		xdotmu[i] = (eulersy[2*i]-eulersy[2*i-1]) / Δ
	end

	if rmfl
		xmu = xmu[2:end-1,:]; xvar = xvar[2:end-1,:]
		xdotmu = xdotmu[2:end-1,:]
		x = x[2:end-1,:]
	end
	x, xmu, xvar, xdotmu, nothing

end


function construct_parsets(numspec\item Adaptive Gaussian processes - Too slow (R package), used constrained optimisation for GPs instead.ies, maxinter, interactions::AbstractArray; selfinter=false, gpsubtract=true)
	allparents = collect(1:numspecies)
	parsets::Array{Parset,1} = []
	count::Int = 0

	if selfinter
		for i = 1:numspecies
			for k = 1:maxinter
				for l in Combinatorics.combinations(allparents, k)
					for m in Iterators.product(Iterators.repeated(interactions,k)...)
						if i == l[1] && length(l) == 1
							continue
						end
						count += 1
						push!(parsets, Parset(count,i, k, l, collect(m), [], 0.0, 0.0, 0.0, 0.0, 0.0))
					end
				end
			end
		end
	else
		for i = 1:numspecies
			for k = 1:maxinter
				for l in Combinatorics.combinations(filter(e -> e ≠ i, allparents), k)
					for m in Iterators.product(Iterators.repeated(interactions,k)...)
						count += 1
						push!(parsets, Parset(count, i, k, l, collect(m), [], 0.0, 0.0, 0.0, 0.0, 0.0))
					end
				end
			end
		end
	end
	reshape(parsets,(:,numspecies))
end

function construct_parsets(numspecies, maxinter, interactions::Void; selfinter=false, gpsubtract=true)
	allparents = collect(1:numspecies)
	gpparsets::Array{GPparset,1} = []
	count::Int = 0

	if selfinter && gpsubtract
		for i = 1:numspecies
			for k = 1:maxinter
				for l in Combinatorics.combinations(allparents, k)
					if i == l[1] && length(l) == 1
						continue
					end
					count += 1
					push!(gpparsets, GPparset(count, i, k, l, 0.0, 0.0, 0.0, 0.0, 0.0))
				end
			end
		end
	elseif !selfinter && gpsubtract
		for i = 1:numspecies
			for k = 1:maxinter
				for l in Combinatorics.combinations(filter(e -> e ≠ i, allparents), k)
					count += 1
					push!(gpparsets, GPparset(count, i, k, l, 0.0, 0.0, 0.0, 0.0, 0.0))
				end
			end
		end
	elseif selfinter && !gpsubtract
		for i = 1:numspecies
			for k = 1:maxinter
				for l in Combinatorics.combinations(filter(e -> e ≠ i, allparents), k)
					count += 1
					parents = [i;l]
					push!(gpparsets, GPparset(count, i, k+1, parents, 0.0, 0.0, 0.0, 0.0, 0.0))
				end
			end
		end
	else
		error("Invalid combination of selfinter and gpsubtract keywords.")
	end
	reshape(gpparsets,(:,numspecies))
end


function construct_ode(topology, fixparm, xmu, xdotmu)
	initp::Array{Float64,1} = [1.0]
	lowerb::Array{Float64,1} = [0.0]
	upperb::Array{Float64,1} = [5.0]

	for parent in enumerate(topology.parents)
		push!(initp,2.0,1.0)
		push!(lowerb,0.1,0.0)
		push!(upperb,5.0,4.0)
	end

	function odefunc(p)
		# !This function does not work for species without parents!
		basis = fixparm[1,topology.speciesnum] .- fixparm[2,topology.speciesnum] .* xmu[:,topology.speciesnum]
		fact = ones(Float64, size(xmu,1))
		frepr = ones(Float64, size(xmu,1))
		it::Int = 0

		for parent in enumerate(topology.parents)
			if topology.intertype[parent[1]] == :Activation
				if it == 0
					fact -= ones(Float64, size(xmu,1))
				end
				it += 2
				fact += (xmu[:,parent[2]] .^ p[it]) ./ (p[it+1] ^ p[it] + xmu[:,parent[2]] .^ p[it])
			end
		end

		for parent in enumerate(topology.parents)
			if topology.intertype[parent[1]] == :Repression
				it += 2
				frepr .*= (1.0 ./ (1.0 .+ (xmu[:,parent[2]] ./ p[it+1]) .^ p[it]))
			end
		end

		sum(((basis .+ p[1] .* fact .* frepr ) - xdotmu[:,topology.speciesnum]) .^ 2.0)
	end

	odefunc, initp, lowerb, upperb
end

function construct_ode_osc(topology, fixparm, xmu, xdotmu)
	initp::Array{Float64,1} = []
	lowerb::Array{Float64,1} = []
	upperb::Array{Float64,1} = []

	for parent in enumerate(topology.parents)
		push!(initp,1.0,1.0,1.0)
		# push!(initp,rand(Distributions.Uniform(0.5,4.0), 1), rand(Distributions.Uniform(0.7,5.0), 1), rand(Distributions.Uniform(0.2,3.0), 1))
		push!(lowerb,0.5,0.7,0.2)
		push!(upperb,4.0,5.0,3.0)
	end

	function odefunc(p)
		# !This function does not work for species without parents!
		basis = fixparm[1,topology.speciesnum] .- fixparm[2,topology.speciesnum] .* xmu[:,topology.speciesnum]
		fact = zeros(Float64, size(xmu,1))
		frepr = zeros(Float64, size(xmu,1))
		it::Int = 1

		for parent in enumerate(topology.parents)
			if topology.intertype[parent[1]] == :Activation
				fact += p[it] .* (xmu[:,parent[2]] .^ p[it+1]) ./ (p[it+2] ^ p[it+1] + xmu[:,parent[2]] .^ p[it+1])
				it += 3
			end
		end

		for parent in enumerate(topology.parents)
			if topology.intertype[parent[1]] == :Repression
				frepr .+= p[it] .* (1.0 ./ (1.0 .+ (xmu[:,parent[2]] ./ p[it+2]) .^ p[it+1]))
				it += 3
			end
		end

		sum(((basis .+ fact .+ frepr ) - xdotmu[:,topology.speciesnum]) .^ 2.0)
	end

	odefunc, initp, lowerb, upperb
end


function optimise_params!(topology::Parset, fixparm, xmu, xdotmu, osc)
	if osc
		f, initial, lower, upper = construct_ode_osc(topology, fixparm, xmu, xdotmu)
	else
		f, initial, lower, upper = construct_ode(topology, fixparm, xmu, xdotmu)
	end

	results = Optim.optimize(f, initial, lower, upper, Optim.Fminbox{Optim.NelderMead}())
	n = size(xmu,1)
	topology.params = results.minimizer
	topology.modaic = n * log(results.minimum / n) + 2 * length(results.minimizer)
	topology.modbic = n * log(results.minimum / n) + log(n)*length(results.minimizer)
	topology.dist = results.minimum
end

function optimise_params!(gppar::GPparset, x, y)
	kernel = gkern.RBF(input_dim=gppar.intercount, variance=1, lengthscale=1)
	m = gmodels.GPRegression(x,y,kernel)
	m[:optimize_restarts](num_restarts = 5, verbose=false)
	# m[:plot](plot_density=false)
	gppar.lik = m[:log_likelihood]()
	gppar.modaic =  2 * gppar.intercount - 2 * gppar.lik
	gppar.modbic = log(size(y)[1]) * gppar.intercount - 2 * gppar.lik
end


function optimise_models!(parsets::Array{Parset,2}, fixparm, xmu, xdotmu, osc; gpsubtract=true)
	@progress "Optimising parameters" for par in parsets
		try
			optimise_params!(par,fixparm,xmu,xdotmu,osc)
		catch err
			if isa(err, DomainError)
				warn("Could not complete optimisation of 1 model.")
				par.params = [NaN]
				par.modaic = Inf
				par.dist = Inf
			else
				error("Non - DomainError in optimisation.")
			end
		end
	end
end

function optimise_models!(parsets::Array{GPparset,2}, fixparm, xmu, xdotmu, osc; gpsubtract=true)
	if gpsubtract
		@progress "Optimising GPs" for gppar in parsets
			X = xmu[:,gppar.parents]
			Y = reshape(xdotmu[:,gppar.speciesnum] - fixparm[1,gppar.speciesnum] + fixparm[2,gppar.speciesnum] .* xmu[:,gppar.speciesnum],(:,1))
			optimise_params!(gppar,X,Y)
		end
	else
		@progress "Optimising GPs" for gppar in parsets
			X = xmu[:,gppar.parents]
			Y = reshape(xdotmu[:,gppar.speciesnum],(:,1))
			optimise_params!(gppar,X,Y)
		end
	end
end


function weight_models!(parsets)
	for i = 1:size(parsets,2)
		minaic = minimum(p.modaic for p in parsets[:,i])
		minbic = minimum(p.modbic for p in parsets[:,i])
		denomsumaic = sum(exp((-p.modaic + minaic) / 2) for p in parsets[:,i])
		denomsumbic = sum(exp((-p.modbic + minbic) / 2) for p in parsets[:,i])

		for j in parsets[:,i]
			j.aicweight = exp((-j.modaic + minaic) / 2) / denomsumaic
			j.bicweight = exp((-j.modbic + minbic) / 2) / denomsumbic
		end
	end
end


function weight_edges(parsets::Array{Parset,2}, interactions::AbstractArray)
	# Create edgeweight array with columns:
	# Target, Source, Interaction, Weight

	numsp = size(parsets,2)
	numint = length(interactions)
	interdict = map(reverse, Dict(enumerate(interactions)))
	edgeweights = hcat(repeat(collect(1:numsp),inner=numsp*numint),
						repeat(collect(1:numsp),outer=numsp,inner=numint),
						repeat(interactions,outer=numsp^2),
						zeros(numint*numsp^2),
						zeros(numint*numsp^2))

	for parset in parsets
		for (i, parent) in enumerate(parset.parents)
			row = (parset.speciesnum - 1) * numsp * numint + (parent-1) * numint + interdict[parset.intertype[i]]
			edgeweights[row,4] += parset.aicweight
			edgeweights[row,5] += parset.bicweight
		end
	end
	edgeweights
end

function weight_edges(parsets::Array{GPparset,2}, interactions::Void)
	# Create edgeweight array with columns:
	# Target, Source, Weight

	numsp = size(parsets,2)
	edgeweights = hcat(repeat(collect(1:numsp),inner=numsp),
						repeat(collect(1:numsp),outer=numsp),
						zeros(numsp^2),
						zeros(numsp^2))

	for parset in parsets
		for (i, parent) in enumerate(parset.parents)
			row = (parset.speciesnum - 1) * numsp + (parent-1) + 1
			edgeweights[row,3] += parset.aicweight
			edgeweights[row,4] += parset.bicweight
		end
	end
	edgeweights
end


function get_true_ranks(trueparents, parsets::Array{Parset,2})
	# Create ranks array with columns representing each specie and rows:
	# Specie Number, ID, Distance, AIC, BIC, Distrank, AICrank, BICrank, AICweight, BICweight

	ranks = []
	for tp in trueparents
		for parset in parsets[:, tp.speciesnum]
			if tp.parents == parset.parents && tp.intertype == parset.intertype
				distrank = find(sort([i.dist for i in parsets[:,tp.speciesnum]]) .== parset.dist)
				aicrank = find(sort([i.modaic for i in parsets[:,tp.speciesnum]]) .== parset.modaic)
				bicrank = find(sort([i.modbic for i in parsets[:,tp.speciesnum]]) .== parset.modbic)
				if tp.speciesnum == 1
					ranks = [parset.speciesnum, parset.id, parset.dist, parset.modaic, parset.modbic,
								distrank, aicrank, bicrank, parset.aicweight, parset.bicweight]
				else
					ranks = hcat(ranks, [parset.speciesnum, parset.id, parset.dist, parset.modaic, parset.modbic,
											distrank, aicrank, bicrank, parset.aicweight, parset.bicweight])
				end
				break
			end
		end
	end
	ranks
end

function get_true_ranks(trueparents, parsets::Array{GPparset,2})
	# Create ranks array with columns representing each specie and rows:
	# Specie Number, ID, Log_likelihood, AIC, BIC, Likrank, AICrank, BICrank, AICweight, BICweight

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


function get_best_id(parsets::Array{Parset,2})
	# Create array with top scoring models. Columns representing each specie and rows:
	# Specie Number, ID of min dist mod, ID of min aic mod, ID of min bic mod, Dist of min dist mod, AIC of min aic mod, BIC of min bic mod

	bestlist = []
	for j = 1:size(parsets,2)
		row = find([i.dist for i in parsets[:,j]] .== minimum(i.dist for i in parsets[:,j]))[1]
		distid = parsets[row,j].id
		row = find([i.modaic for i in parsets[:,j]] .== minimum(i.modaic for i in parsets[:,j]))[1]
		aicid = parsets[row,j].id
		row = find([i.modbic for i in parsets[:,j]] .== minimum(i.modbic for i in parsets[:,j]))[1]
		bicid = parsets[row,j].id
		if j == 1
			bestlist = [j, distid, aicid, bicid, parsets[distid].dist, parsets[aicid].modaic, parsets[bicid].modbic]
		else
			bestlist = hcat(bestlist,[j, distid, aicid, bicid, parsets[distid].dist, parsets[aicid].modaic, parsets[bicid].modbic])
		end
	end
	bestlist
end

function get_best_id(parsets::Array{GPparset,2})
	# Create array with top scoring models. Columns representing each specie and rows:
	# Specie Number, ID of max log_lik mod, ID of min aic mod, ID of min bic mod, Log_lik of max log_lik mod, AIC of min aic mod, BIC of min bic mod

	bestlist = []
	for j = 1:size(parsets,2)
		row = find([i.lik for i in parsets[:,j]] .== maximum(i.lik for i in parsets[:,j]))[1]
		distid = parsets[row,j].id
		row = find([i.modaic for i in parsets[:,j]] .== minimum(i.modaic for i in parsets[:,j]))[1]
		aicid = parsets[row,j].id
		row = find([i.modbic for i in parsets[:,j]] .== minimum(i.modbic for i in parsets[:,j]))[1]
		bicid = parsets[row,j].id
		if j == 1
			bestlist = [j, distid, aicid, bicid, parsets[distid].lik, parsets[aicid].modaic, parsets[bicid].modbic]
		else
			bestlist = hcat(bestlist,[j, distid, aicid, bicid, parsets[distid].lik, parsets[aicid].modaic, parsets[bicid].modbic])
		end
	end
	bestlist
end


function edgesummary(edgeweights,trueparents)
	othersum = [0.0, 0.0]
	truedges = Array{Any,2}(1,1)
	first = true
	if size(edgeweights,1) == length(trueparents)^2
		for i in 1:size(edgeweights,1)
			if edgeweights[i,2] in trueparents[convert(Int,edgeweights[i,1])].parents
				if first
					truedges = edgeweights[i,:]'
					first = false
				else
					truedges = vcat(truedges, edgeweights[i,:]')
				end
			else
				othersum .+= edgeweights[i,[3,4]]
			end
		end
	else
		for i in 1:size(edgeweights,1)
			thispset = trueparents[convert(Int,edgeweights[i,1])]
			if edgeweights[i,2] in thispset.parents &&
					[edgeweights[i,3]] == thispset.intertype[find(thispset.parents .== edgeweights[i,2])]
				if first
					truedges = reshape(edgeweights[i,:],(1,:))
					first = false
				else
					truedges = vcat(truedges, reshape(edgeweights[i,:],(1,:)))
				end
			else
				othersum .+= edgeweights[i,[4,5]]
			end
		end
	end
	truedges, othersum
end


function metricdata(edgeweights,trueparents)
	truth = Vector{Bool}(0)
	indx = 0
	if size(edgeweights,1) == length(trueparents)^2
		for i in 1:size(edgeweights,1)
			if edgeweights[i,2] in trueparents[convert(Int,edgeweights[i,1])].parents
				push!(truth,true)
			else
				push!(truth,false)
			end
		end
	else
		indx = 1
		for i in 1:size(edgeweights,1)
			thispset = trueparents[convert(Int,edgeweights[i,1])]
			if edgeweights[i,2] in thispset.parents &&
					[edgeweights[i,3]] == thispset.intertype[find(thispset.parents .== edgeweights[i,2])]
				push!(truth,true)
			else
				push!(truth,false)
			end
		end
	end
	truth, edgeweights[:,[3+indx,4+indx]]
end


end
