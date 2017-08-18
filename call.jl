push!(LOAD_PATH, "/cluster/home/ld2113/work/Final-Project/")

using GPinf


tspan = (0.0,20.0)
δt = 0.5
σ = 0.0
maxinter = 2
slfint = true
gpsbt = false

# interactions = [:Activation, :Repression]
interactions = nothing

osc = false

################################################################################

if osc
	function odesys(t,u,du)
		du[1] = 0.2 - 0.9*u[1] + 2.0*((u[5]^5.0)/(1.5^5.0+u[5]^5.0))
		du[2] = 0.2 - 0.9*u[2] + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0))
		du[3] = 0.2 - 0.7*u[3] + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0))
		du[4] = 0.2 - 1.5*u[4] + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0)) + 2.0*(1.0/(1.0+(u[3]/1.5)^5.0))
		du[5] = 0.2 - 1.5*u[5] + 2.0*((u[4]^5.0)/(1.5^5.0+u[4]^5.0)) + 2.0*(1.0/(1.0+(u[2]/1.5)^3.0))
	end
	fixparm = [0.2 0.2 0.2 0.2 0.2; 0.9 0.9 0.7 1.5 1.5]
	if interactions == nothing && !gpsbt
		trueparents = [Parset(0, 1, 2, [1, 5], [:Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 2, 2, [2, 1], [:Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 3, 2, [3, 1], [:Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 4, 3, [4, 1, 3], [:Activation, :Repression], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 5, 3, [5, 2, 4], [:Repression, :Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0)]
	else
		trueparents = [Parset(0, 1, 1, [5], [:Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 2, 1, [1], [:Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 3, 1, [1], [:Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 4, 2, [1, 3], [:Activation, :Repression], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 5, 2, [2, 4], [:Repression, :Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0)]
	end
else
	function odesys(t,u,du)
		du[1] = 0.1-0.4*u[1]+2.0*((u[5]^2.0)/(1.5^2.0+u[5]^2.0))
		du[2] = 0.2-0.4*u[2]+1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[5]/2.0)^1.0))
		du[3] = 0.2-0.4*u[3]+2.0*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))
		du[4] = 0.4-0.1*u[4]+1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[3]/1.0)^2.0))
		du[5] = 0.3-0.3*u[5]+2.0*((u[4]^2.0)/(1.0^2.0+u[4]^2.0))*(1.0/(1.0+(u[2]/0.5)^3.0))
	end
	fixparm = [0.1 0.2 0.2 0.4 0.3; 0.4 0.4 0.4 0.1 0.3]
	if interactions == nothing && !gpsbt
		trueparents = [Parset(0, 1, 2, [1, 5], [:Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 2, 3, [2, 1, 5], [:Activation, :Repression], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 3, 2, [3, 1], [:Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 4, 3, [4, 1, 3], [:Activation, :Repression], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 5, 3, [5, 2, 4], [:Repression, :Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0)]
	else
		trueparents = [Parset(0, 1, 1, [5], [:Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 2, 2, [1, 5], [:Activation, :Repression], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 3, 1, [1], [:Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 4, 2, [1, 3], [:Activation, :Repression], [], 0.0, 0.0, 0.0, 0.0, 0.0),
						Parset(0, 5, 2, [2, 4], [:Repression, :Activation], [], 0.0, 0.0, 0.0, 0.0, 0.0)]
	end
end

x,y = simulate(odesys, tspan, δt; noise=σ)
# x = x[21:41,:]
# y = y[21:41,:]
numspecies = size(y,2)

xmu, xvar, xdotmu, xdotvar = interpolate_single(x, y)

parsets = construct_parsets(numspecies, maxinter, interactions; selfinter=slfint, gpsubtract=gpsbt)

optimise_models!(parsets, fixparm, xmu, xdotmu, osc, gpsubtract=gpsbt)

weight_models!(parsets)

edgeweights = weight_edges(parsets, interactions)

ranks = get_true_ranks(trueparents, parsets)

bestmodels = get_best_id(parsets)
