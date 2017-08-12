import DifferentialEquations, PyCall, PyPlot, Distributions, Optim, Combinatorics

PyCall.@pyimport GPy.kern as gkern
PyCall.@pyimport GPy.models as gmodels


mutable struct GPparset
	speciesnum::Int
	intercount::Int
	parents::Array{Int, 1}
	lik::Float64
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


function construct_gpparsets(numspecies, maxinter; selfinter=false)
	allparents = collect(1:numspecies)
	gpparsets::Array{GPparset,1} = []

	if selfinter
		for i = 1:numspecies
			for k = 1:maxinter
				for l in Combinatorics.combinations(allparents, k)
					push!(gpparsets, GPparset(i, k, l, 0.0))
				end
			end
		end

	else
		for i = 1:numspecies
			for k = 1:maxinter
				for l in Combinatorics.combinations(filter(e -> e ≠ i, allparents), k)
					push!(gpparsets, GPparset(i, k, l, 0.0))
				end
			end
		end
	end

	reshape(gpparsets,(:,numspecies))
end

function compute_lik!(gppar::GPparset,x,y)



function get_all_lik!(gpparsets,y,xdotmu)
	for gppar in gpparsets
		compute_lik!(gppar,x,y)




################################################################################


x, y = simulate()

const numspecies = size(y,2)
const maxinter = 2
const interactions = [:Activation, :Repression]
const fixparm = [0.1 0.2 0.2 0.4 0.3; 0.4 0.4 0.4 0.1 0.3]

xmu, xvar, xdotmu, xdotvar = interpolate(x, y)

gpparsets = construct_gpparsets(numspecies, maxinter; selfinter=true)
