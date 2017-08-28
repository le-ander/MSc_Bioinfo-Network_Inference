module GPinf

using Juno
import DifferentialEquations, PyCall, Distributions, Optim, Combinatorics, NetworkInference, ScikitLearn
# import PyPlot

ScikitLearn.@sk_import metrics: (average_precision_score, precision_recall_curve,
									roc_auc_score, roc_curve)

PyCall.@pyimport GPy.kern as gkern
PyCall.@pyimport GPy.models as gmodels
PyCall.@pyimport GPy.util.multioutput as gmulti

export Parset, GPparset, TParset,
		datasettings, simulate, interpolate,
		construct_parsets, optimise_models!, weight_models!,
		weight_edges, get_true_ranks, get_best_id, edgesummary, performance,
		networkinference


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
	params::Array{Float64, 1}
	lik::Float64
	modaic::Float64
	modbic::Float64
	aicweight::Float64
	bicweight::Float64
end

struct TParset
	speciesnum::Int
	intercount::Int
	parents::Array{Int, 1}
	intertype::Array{Symbol,1}
end


function datasettings(srcset::Symbol, interclass, usefix)
	initp::Vector{Float64} = []
	lowerb::Vector{Float64} = []
	upperb::Vector{Float64} = []
	repinit::Vector{Float64} = []
	replow::Vector{Float64} = []
	rephigh::Vector{Float64} = []
	function oscodesys(t,u,du)
		du[1] = 0.2 - 0.9*u[1] + 2.0*((u[5]^5.0)/(1.5^5.0+u[5]^5.0))
		du[2] = 0.2 - 0.9*u[2] + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0))
		du[3] = 0.2 - 0.7*u[3] + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0))
		du[4] = 0.2 - 1.5*u[4] + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0)) + 2.0*(1.0/(1.0+(u[3]/1.5)^5.0))
		du[5] = 0.2 - 1.5*u[5] + 2.0*((u[4]^5.0)/(1.5^5.0+u[4]^5.0)) + 2.0*(1.0/(1.0+(u[2]/1.5)^3.0))
	end
	function oscsdesys(t, u, du)
		du[1,1] = √(0.2 + 2.0*((u[5]^5.0)/(1.5^5.0+u[5]^5.0))); du[1,6] = -√(0.9*u[1]);
		du[2,2] = √(0.2 + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0))); du[2,7] = -√(0.9*u[2]);
		du[3,3] = √(0.2 + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0))); du[3,8] = -√(0.7*u[3]);
		du[4,4] = √(0.2 + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0)) + 2.0*(1.0/(1.0+(u[3]/1.5)^5.0))); du[4,9] = -√(1.5*u[4]);
		du[5,5] = √(0.2 + 2.0*((u[4]^5.0)/(1.5^5.0+u[4]^5.0)) + 2.0*(1.0/(1.0+(u[2]/1.5)^3.0))); du[5,10] = -√(1.5*u[5]);
	end
	function linodesys(t,u,du)
		du[1] = 0.1-0.4*u[1]+2.0*((u[5]^2.0)/(1.5^2.0+u[5]^2.0))
		du[2] = 0.2-0.4*u[2]+1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[5]/2.0)^1.0))
		du[3] = 0.2-0.4*u[3]+2.0*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))
		du[4] = 0.4-0.1*u[4]+1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[3]/1.0)^2.0))
		du[5] = 0.3-0.3*u[5]+2.0*((u[4]^2.0)/(1.0^2.0+u[4]^2.0))*(1.0/(1.0+(u[2]/0.5)^3.0))
	end
	function linsdesys(t, u, du)
		du[1,1] = √(0.1 + 2.0*((u[5]^2.0)/(1.5^2.0+u[5]^2.0))); du[1,6] = -√(0.4*u[1]);
		du[2,2] = √(0.2 + 1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[5]/2.0)^1.0))); du[2,7] = -√(0.4*u[2]);
		du[3,3] = √(0.2 + 2.0*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))); du[3,8] = -√(0.4*u[3]);
		du[4,4] = √(0.4 + 1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[3]/1.0)^2.0))); du[4,9] = -√(0.1*u[4]);
		du[5,5] = √(0.3 + 2.0*((u[4]^2.0)/(1.0^2.0+u[4]^2.0))*(1.0/(1.0+(u[2]/0.5)^3.0))); du[5,10] = -√(0.3*u[5]);
	end
	if srcset == :osc
		if usefix
			fixparm = [0.2 0.2 0.2 0.2 0.2; 0.9 0.9 0.7 1.5 1.5]
			if interclass == :mult
				initp = [1.0]
				lowerb = [0.5]
				upperb = [4.0]
				repinit = [1.0,1.0]
				replow = [0.7,0.2]
				rephigh = [5.0,3.0]
			elseif interclass == :add
				initp = []
				lowerb = []
				upperb = []
				repinit = [1.0,1.0,1.0]
				replow = [0.5,0.7,0.2]
				rephigh = [4.0,5.0,3.0]
			else
				initp = []
				lowerb = []
				upperb = []
				repinit = []
				replow = []
				rephigh = []
			end
		else
			fixparm = []
			if interclass == :mult
				initp =  [0.5,1.0,1.0]
				lowerb = [0.1,0.5,0.5]
				upperb = [2.0,2.0,4.0]
				repinit = [1.0,1.0]
				replow = [0.7,0.2]
				rephigh = [5.0,3.0]
			elseif interclass == :add
				initp =  [0.5,1.0]
				lowerb = [0.1,0.5]
				upperb = [2.0,2.0]
				repinit = [1.0,1.0,1.0]
				replow = [0.5,0.7,0.2]
				rephigh = [4.0,5.0,3.0]
			else
				initp = []
				lowerb = []
				upperb = []
				repinit = []
				replow = []
				rephigh = []
			end
		end

		if interclass == nothing
			trueparents = [TParset(1, 2, [1, 5], [:Activation]),
							TParset(2, 2, [2, 1], [:Activation]),
							TParset(3, 2, [3, 1], [:Activation]),
							TParset(4, 3, [4, 1, 3], [:Activation, :Repression]),
							TParset(5, 3, [5, 2, 4], [:Repression, :Activation])]
		else
			trueparents = [TParset(1, 1, [5], [:Activation]),
							TParset(2, 1, [1], [:Activation]),
							TParset(3, 1, [1], [:Activation]),
							TParset(4, 2, [1, 3], [:Activation, :Repression]),
							TParset(5, 2, [2, 4], [:Repression, :Activation])]
		end

	elseif srcset == :lin
		if usefix
			fixparm = [0.1 0.2 0.2 0.4 0.3; 0.4 0.4 0.4 0.1 0.3]
			if interclass == :mult
				initp = [1.0]
				lowerb = [0.0]
				upperb = [5.0]
				repinit = [2.0,1.0]
				replow = [0.1,0.0]
				rephigh = [5.0,4.0]
			elseif interclass == :add
				initp = []
				lowerb = []
				upperb = []
				repinit = [1.0,2.0,1.0]
				replow = [0.0,0.1,0.0]
				rephigh = [5.0,5.0,4.0]
			else
				initp = []
				lowerb = []
				upperb = []
				repinit = []
				replow = []
				rephigh = []
			end
		else
			fixparm = []
			if interclass == :mult
				initp =  [0.5,1.0,1.0]
				lowerb = [0.1,0.5,0.0]
				upperb = [2.0,2.0,5.0]
				repinit = [2.0,1.0]
				replow = [0.1,0.0]
				rephigh = [5.0,4.0]
			elseif interclass == :add
				initp =  [0.5,1.0]
				lowerb = [0.1,0.5]
				upperb = [2.0,2.0]
				repinit = [1.0,2.0,1.0]
				replow = [0.0,0.1,0.0]
				rephigh = [5.0,5.0,4.0]
			else
				initp = []
				lowerb = []
				upperb = []
				repinit = []
				replow = []
				rephigh = []
			end
		end

		if interclass == nothing
			trueparents = [TParset(1, 2, [1, 5], [:Activation]),
							TParset(2, 3, [2, 1, 5], [:Activation, :Repression]),
							TParset(3, 2, [3, 1], [:Activation]),
							TParset(4, 3, [4, 1, 3], [:Activation, :Repression]),
							TParset(5, 3, [5, 2, 4], [:Repression, :Activation])]
		else
			trueparents = [TParset(1, 1, [5], [:Activation]),
							TParset(2, 2, [1, 5], [:Activation, :Repression]),
							TParset(3, 1, [1], [:Activation]),
							TParset(4, 2, [1, 3], [:Activation, :Repression]),
							TParset(5, 2, [2, 4], [:Repression, :Activation])]
		end

	elseif srcset == :gnw
		fixparm = []
		if interclass == :mult
			initp =  [0.5,1.0,1.0]
			lowerb = [0.1,0.5,0.0]
			upperb = [2.0,2.0,5.0]
			repinit = [2.0,1.0]
			replow = [0.1,0.0]
			rephigh = [5.0,4.0]
		elseif interclass == :add
			initp =  [0.5,1.0]
			lowerb = [0.1,0.5]
			upperb = [2.0,2.0]
			repinit = [1.0,2.0,1.0]
			replow = [0.0,0.1,0.0]
			rephigh = [5.0,5.0,4.0]
		else
			initp = []
			lowerb = []
			upperb = []
			repinit = []
			replow = []
			rephigh = []
		end

		if interclass == nothing
			trueparents = [TParset(1, 2, [1, 3], [:Repression]),
							TParset(2, 2, [2, 1], [:Repression]),
							TParset(3, 1, [3], []),
							TParset(4, 2, [4, 6], [:Repression, :Repression]),
							TParset(5, 3, [5, 1, 3], [:Activation, :Repression]),
							TParset(6, 1, [6], []),
							TParset(7, 3, [7, 5, 10], [:Repression, :Repression]),
							TParset(8, 2, [8, 7], [:Repression]),
							TParset(9, 2, [9, 4], [:Repression]),
							TParset(10, 1, [10], [])]
		else
			trueparents = [TParset(1, 1, [3], [:Repression]),
							TParset(2, 1, [1], [:Repression]),
							TParset(3, 0, [], []),
							TParset(4, 2, [1, 6], [:Repression, :Repression]),
							TParset(5, 2, [1, 3], [:Activation, :Repression]),
							TParset(6, 0, [], []),
							TParset(7, 2, [5, 10], [:Repression, :Repression]),
							TParset(8, 1, [7], [:Repression]),
							TParset(9, 1, [4], [:Repression]),
							TParset(10, 0, [], [])]
		end
	end
	if srcset == :osc
		return oscodesys, oscsdesys, fixparm, trueparents, hcat(initp, lowerb, upperb), hcat(repinit, replow, rephigh)
	elseif srcset == :lin
		return linodesys, linsdesys, fixparm, trueparents, hcat(initp, lowerb, upperb), hcat(repinit, replow, rephigh)
	elseif srcset == :gnw
		return nothing, nothing, fixparm, trueparents, hcat(initp, lowerb, upperb), hcat(repinit, replow, rephigh)
	end
end

function readgenes(path::String)
	xy = readdlm(open(path); skipstart = 1)
	x = xy[:,1:1]
	y = xy[:,2:end]
	x, y
end


function simulate(odesys, sdesys, numspecies, tspan, step, noise::Float64)
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

function simulate(odesys, sdesys, numspecies, tspan, step, noise::Symbol)
	A = hcat(eye(numspecies),-eye(numspecies))
	sparse(A)

	u0 = [1.0;0.5;1.0;0.5;0.5] # Define initial conditions

	prob = DifferentialEquations.SDEProblem(odesys,sdesys,u0,tspan,noise_rate_prototype=A)
	sol = DifferentialEquations.solve(prob, dt=0.001, saveat=step)
	x = reshape(sol.t,(length(sol.t),1))
	y = hcat(sol.u...)'
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
		# m[:plot](plot_density=false)

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


function construct_parsets(numspecies, maxinter, fixparm, interclass::Symbol)
	allparents = collect(1:numspecies)
	parsets::Array{Parset,1} = []
	count::Int = 0
	interactions = [:Activation, :Repression]

	for i = 1:numspecies
		count += 1
		push!(parsets, Parset(count, i, 0, [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0))
	end

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

	return reshape(parsets,(:,numspecies))

	"""
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

	end
	"""
end

function construct_parsets(numspecies, maxinter, fixparm, interclass::Void)
	allparents = collect(1:numspecies)
	gpparsets::Array{GPparset,1} = []
	count::Int = 0

	for i = 1:numspecies
		for k = 0:maxinter
			for l in Combinatorics.combinations(filter(e -> e ≠ i, allparents), k)
				count += 1
				parents = [i;l]
				push!(gpparsets, GPparset(count, i, k+1, parents, [], 0.0, 0.0, 0.0, 0.0, 0.0))
			end
		end
	end

	return reshape(gpparsets,(:,numspecies))

	"""
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
		for i = 0:numspecies
			for k = 1:maxinter
				for l in Combinatorics.combinations(filter(e -> e ≠ i, allparents), k)
					count += 1
					push!(gpparsets, GPparset(count, i, k, l, [], 0.0, 0.0, 0.0, 0.0, 0.0))
				end
			end
		end
	elseif selfinter && !gpsubtract
		...
	else
		error("Invalid combination of selfinter and gpsubtract keywords.")
	end
	"""
end


function construct_ode(topology, fixparm, xmu, xdotmu, interclass)
	numspecies = size(xmu,2)
	if interclass == :mult
		function multodefunc(p)
			if fixparm != []
				basis = fixparm[1,topology.speciesnum] .- fixparm[2,topology.speciesnum] .* xmu[:,topology.speciesnum]
				it = 0
			else
				basis = p[1] .- p[2] .* xmu[:,topology.speciesnum]
				it = 2
			end

			if topology.intercount == 0
				return sum((basis .- xdotmu[:,topology.speciesnum]) .^ 2.0)
			end

			fact = ones(Float64, size(xmu,1))
			frepr = ones(Float64, size(xmu,1))

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

			sum(((basis .+ p[1] .* fact .* frepr ) .- xdotmu[:,topology.speciesnum]) .^ 2.0)
		end
		return multodefunc
	elseif interclass == :add
		function addodefunc(p)
			if fixparm != []
				basis = fixparm[1,topology.speciesnum] .- fixparm[2,topology.speciesnum] .* xmu[:,topology.speciesnum]
				it = 1
			else
				basis = p[1] .- p[2] .* xmu[:,topology.speciesnum]
				it = 3
			end

			if topology.intercount == 0
				return sum((basis .- xdotmu[:,topology.speciesnum]) .^ 2.0)
			end

			fact = zeros(Float64, size(xmu,1))
			frepr = zeros(Float64, size(xmu,1))

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

			sum(((basis .+ fact .+ frepr ) .- xdotmu[:,topology.speciesnum]) .^ 2.0)
		end
		return addodefunc
	end
end


function optimise_params!(topology::Parset, fixparm, xmu, xdotmu, interclass, initial, lower, upper, prmrep)
	f = construct_ode(topology, fixparm, xmu, xdotmu, interclass)
	n = size(xmu,1)
	if topology.intercount == 0 && fixparm != []
		topology.dist = f([])
		topology.modaic = n * log(topology.dist / n) + 2 * 0
		topology.modbic = n * log(topology.dist / n) + log(n) * 0
	else
		for parent in topology.parents
			initial = vcat(initial,prmrep[:,1])
			lower = vcat(lower,prmrep[:,2])
			upper = vcat(upper,prmrep[:,3])
		end
		results = Optim.optimize(f, initial[:], lower[:], upper[:], Optim.Fminbox{Optim.NelderMead}())
		topology.params = results.minimizer
		topology.modaic = n * log(results.minimum / n) + 2 * length(results.minimizer)
		topology.modbic = n * log(results.minimum / n) + log(n)*length(results.minimizer)
		topology.dist = results.minimum
	end
end

function optimise_params!(gppar::GPparset, x, y)
	kernel = gkern.RBF(input_dim=gppar.intercount, variance=1, lengthscale=1)
	m = gmodels.GPRegression(x,y,kernel)
	m[:optimize_restarts](num_restarts = 5, verbose=false)
	# m[:plot](plot_density=false)
	gppar.params = m[:param_array]
	gppar.lik = m[:log_likelihood]()
	gppar.modaic =  2 * gppar.intercount - 2 * gppar.lik
	gppar.modbic = log(size(y)[1]) * gppar.intercount - 2 * gppar.lik
end


function optimise_models!(parsets::Array{Parset,2}, fixparm, xmu, xdotmu, interclass::Symbol, prmrng, prmrep)
	initp = prmrng[:,1:1]
	lowerb = prmrng[:,2:2]
	upperb = prmrng[:,3:3]
	@progress "Optimising ODEs" for par in parsets
		try
			optimise_params!(par, fixparm, xmu, xdotmu, interclass, initp, lowerb, upperb, prmrep)
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

function optimise_models!(parsets::Array{GPparset,2}, fixparm, xmu, xdotmu, interclass::Void, prmrng, prmrep)
	@progress "Optimising GPs" for gppar in parsets
		X = xmu[:,gppar.parents]
		Y = reshape(xdotmu[:,gppar.speciesnum],(:,1))
		optimise_params!(gppar,X,Y)
	end
	"""
	if fixparm != []
		@progress "Optimising GPs" for gppar in parsets
			X = xmu[:,gppar.parents]
			Y = reshape(xdotmu[:,gppar.speciesnum] - fixparm[1,gppar.speciesnum] + fixparm[2,gppar.speciesnum] .* xmu[:,gppar.speciesnum],(:,1))
			optimise_params!(gppar,X,Y)
		end
	else
		...
	end
	"""
	return
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


function weight_edges(parsets::Array{Parset,2}, suminter, interclass::Symbol)
	# Create edgeweight array with columns:
	# Target, Source, Interaction, Weight
	interactions = [:Activation, :Repression]
	if suminter
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

	else
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
	end
	edgeweights
end

function weight_edges(parsets::Array{GPparset,2}, suminter, interclass::Void)
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


function get_true_ranks(trueparents, parsets::Array{Parset,2}, suminter)
	# Create ranks array with columns representing each specie and rows:
	# Specie Number, ID, Distance, AIC, BIC, Distrank, AICrank, BICrank, AICweight, BICweight
	if suminter
		warn("Summed interactions (Activ. + Repr.) are not taken into account during ranking.")
	end
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

function get_true_ranks(trueparents, parsets::Array{GPparset,2}, suminter)
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


function get_best_id(parsets::Array{Parset,2}, suminter)
	# Create array with top scoring models. Columns representing each specie and rows:
	# Specie Number, ID of min dist mod, ID of min aic mod, ID of min bic mod, Dist of min dist mod, AIC of min aic mod, BIC of min bic mod
	if suminter
		warn("Summed interactions (Activ. + Repr.) are not taken into account during ranking.")
	end
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

function get_best_id(parsets::Array{GPparset,2}, suminter)
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


function performance(edgeweights,trueparents)
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
	scores = edgeweights[:,[3+indx,4+indx]]

	# prcurve = precision_recall_curve(truth, scrs[:,1])[1:2]
	# roccurve = roc_curve(truth, scrs[:,1])[1:2]
	# PyPlot.plot(prcurve[2],prcurve[1])
	# PyPlot.plot(roccurve[1],roccurve[2])

	aupr_aic = average_precision_score(truth, scores[:,1])
	aupr_bic = average_precision_score(truth, scores[:,2])
	auroc_aic = roc_auc_score(truth, scores[:,1])
	auroc_bic = roc_auc_score(truth, scores[:,2])

	aupr_aic, aupr_bic, auroc_aic, auroc_bic
end


function networkinference(y, trueparents)
	truth = Vector{Bool}(0)
	number_of_genes = size(y, 2)
	genes = Array{NetworkInference.Gene}(number_of_genes)
	labels = collect(1:number_of_genes)'
	data = vcat(labels,y)

	for i in 1:number_of_genes
		genes[i] = NetworkInference.Gene(data[:,i], "bayesian_blocks", "maximum_likelihood", 10)
	end

	network_analysis = NetworkInference.NetworkAnalysis(NetworkInference.PIDCNetworkInference(), genes)

	output = Array{Any}(length(network_analysis.edges),3)

	for (i,edge) in enumerate(network_analysis.edges)
		output[i,1] = edge.genes[1].name
		output[i,2] = edge.genes[2].name
		output[i,3] = edge.confidence
	end

	for i in 1:size(output,1)
		inthere = false
		edge = [float(output[i,1]), float(output[i,2])]
		for parent in trueparents
			if (edge[1] == parent.speciesnum && edge[2] in parent.parents) || (edge[2] == parent.speciesnum && edge[1] in parent.parents)
				inthere = true
			end
		end
		if inthere
			push!(truth, true)
		else
			push!(truth, false)
		end
	end

	scores = output[:,3]

	aupr = average_precision_score(truth, scores)
	auroc = roc_auc_score(truth, scores)

	# prcurve = precision_recall_curve(truth, scores)[1:2]
	# roccurve = roc_curve(truth, scores)[1:2]
	# PyPlot.plot(prcurve[2],prcurve[1])
	# PyPlot.plot(roccurve[1],roccurve[2])

	# output, truth, scores, aupr, auroc
	output, aupr, auroc
end


end
