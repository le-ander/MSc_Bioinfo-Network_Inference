paral = true

import DifferentialEquations, PyCall, PyPlot, Distributions, Combinatorics

if paral
	@everywhere import Optim
else
	import Optim
end

PyCall.@pyimport GPy.kern as gkern
PyCall.@pyimport GPy.models as gmodels

if paral
	@everywhere mutable struct Parset
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
else
	mutable struct Parset
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
	# y .+= reshape(rand(Distributions.Normal(0, 0.1), length(y)), size(y))
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
		m[:optimize_restarts](num_restarts = 5)
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


function construct_parsets(numspecies, maxinter, interactions; selfinter=false)
	allparents = collect(1:numspecies)
	parsets::Array{Parset,1} = []

	if selfinter
		for i = 1:numspecies
			for k = 1:maxinter
				for l in Combinatorics.combinations(allparents, k)
					for m in Iterators.product(Iterators.repeated(interactions,k)...)
						push!(parsets, Parset(i, k, l, collect(m), [], 0.0, 0.0, 0.0, 0.0, 0.0))
					end
				end
			end
		end

	else
		for i = 1:numspecies
			for k = 1:maxinter
				for l in Combinatorics.combinations(filter(e -> e ≠ i, allparents), k)
					for m in Iterators.product(Iterators.repeated(interactions,k)...)
						push!(parsets, Parset(i, k, l, collect(m), [], 0.0, 0.0, 0.0, 0.0, 0.0))
					end
				end
			end
		end
	end

	reshape(parsets,(:,numspecies))
end

if paral
	@everywhere function construct_ode(topology, fixparm, xmu, xdotmu)
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
					missing = false
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
else
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
					missing = false
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
end


if paral
	function optimise_params(topology, fixparm, xmu, xdotmu)
		f, initial, lower, upper = construct_ode(topology, fixparm, xmu, xdotmu)

		results = Optim.optimize(f, initial, lower, upper, Optim.Fminbox{Optim.NelderMead}())

		n = size(xmu,1)

		modaic = n * log(results.minimum / n) + 2 * length(results.minimizer)
		modbic = n * log(results.minimum / n) + log(n)*length(results.minimizer)
		dist = results.minimum
		[dist, modaic, modbic]
	end
else
	function optimise_params!(topology, fixparm, xmu, xdotmu)
		f, initial, lower, upper = construct_ode(topology, fixparm, xmu, xdotmu)

		results = Optim.optimize(f, initial, lower, upper, Optim.Fminbox{Optim.NelderMead}())

		n = size(xmu,1)

		topology.params = results.minimizer
		topology.modaic = n * log(results.minimum / n) + 2 * length(results.minimizer)
		topology.modbic = n * log(results.minimum / n) + log(n)*length(results.minimizer)
		topology.dist = results.minimum
	end
end

if paral
	function optimise_models!(parsets, fixparm, xmu, xdotmu)
		distlist = SharedArray{Float64}(length(parsets))
		aiclist = SharedArray{Float64}(length(parsets))
		biclist = SharedArray{Float64}(length(parsets))

		@sync @parallel for (i, par) in collect(enumerate(parsets))
			try
				outlist = optimise_params(par,fixparm,xmu,xdotmu)
				distlist[i] = outlist[1]
				aiclist[i] = outlist[2]
				biclist[i] = outlist[3]
			catch
				distlist[i] = Inf
				aiclist[i] = Inf
				biclist[i] = Inf
			end
		end

		for i = 1:length(parsets)
			parsets[i].dist = distlist[i]
			parsets[i].modaic = aiclist[i]
			parsets[i].modbic = biclist[i]
		end
	end
else
	function optimise_models!(parsets, fixparm, xmu, xdotmu)

		@progress "Optimising parameters" for par in parsets
			try
				optimise_params!(par,fixparm,xmu, xdotmu)
			catch
				par.params = [NaN]
				par.modaic = Inf
				par.dist = Inf
			end
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


function weight_edges(parsets, interactions)
	# Create edgeweight array
	# Array columns: Target, Source, Interaction, Weight
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


################################################################################


x, y = simulate()


const numspecies = size(y,2)
const maxinter = 2
const interactions = [:Activation, :Repression]
const fixparm = [0.1 0.2 0.2 0.4 0.3; 0.4 0.4 0.4 0.1 0.3]

xmu, xvar, xdotmu, xdotvar = interpolate(x, y)

parsets = construct_parsets(numspecies, maxinter, interactions, selfinter=false)

optimise_models!(parsets, fixparm, xmu, xdotmu)

weight_models!(parsets)

edgeweights = weight_edges(parsets, interactions)
