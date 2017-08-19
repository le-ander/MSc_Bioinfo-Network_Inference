push!(LOAD_PATH, "/cluster/home/ld2113/work/Final-Project/")
push!(LOAD_PATH, "/media/leander/Daten/Data/Imperial/Projects/Final Project/Final-Project")

using GPinf
import ScikitLearn
ScikitLearn.@sk_import metrics: (average_precision_score, precision_recall_curve,
									roc_auc_score, roc_curve)



osc = false
tspan = (0.0,20.0)
δt = 1.0
σ = 0.1

maxinter = 2
slfint = false
gpsbt = true

# interactions = [:Activation, :Repression]
interactions = nothing

gpnum = 5
# gpnum = nothing

rmfl = true

@show osc; @show tspan; @show δt; @show σ; @show maxinter; @show slfint
@show gpsbt; @show interactions; @show gpnum; @show rmfl

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

xnew, xmu, xvar, xdotmu, xdotvar = interpolate(x, y, rmfl, gpnum)
# xnew2, xmu2, xvar2, xdotmu2, xdotvar2 = interpolate(x, y, rmfl, gpnum)

parsets = construct_parsets(numspecies, maxinter, interactions; selfinter=slfint, gpsubtract=gpsbt)

optimise_models!(parsets, fixparm, xmu, xdotmu, osc, gpsubtract=gpsbt)

weight_models!(parsets)

edgeweights = weight_edges(parsets, interactions)

ranks = get_true_ranks(trueparents, parsets)

bestmodels = get_best_id(parsets)

truedges, othersum = edgesummary(edgeweights,trueparents)

truth, scrs = metricdata(edgeweights,trueparents)

println(average_precision_score(truth, scrs[:,1]))
println(roc_auc_score(truth, scrs[:,1]))
prcurve = precision_recall_curve(truth, scrs[:,1])[1:2]
roccurve = roc_curve(truth, scrs[:,1])[1:2]
PyPlot.plot(prcurve[2],prcurve[1])
PyPlot.plot(roccurve[1],roccurve[2])
