push!(LOAD_PATH, "/cluster/home/ld2113/work/Final-Project/")
push!(LOAD_PATH, "/media/leander/Daten/Data/Imperial/Projects/Final Project/Final-Project")

using GPinf
import ScikitLearn
ScikitLearn.@sk_import metrics: (average_precision_score, precision_recall_curve,
									roc_auc_score, roc_curve)

# Define source data
numspecies = 5
srcset = :osc # :lin :osc :gnw

# Simulate source data
tspan = (0.0,20.0)
δt = 1.0
σ = 0.0 # Std.dev for ode + obs. noise or :sde

# Define possible parent sets
maxinter = 2
interclass = nothing # :add :mult nothing
usefix = true

suminter = false	#ODEonly

gpnum = numspecies # For multioutput gp: how many outputs at once, for single: nothing

rmfl = false

@show numspecies; @show srcset; @show tspan; @show δt; @show σ; @show maxinter;
@show interclass; @show usefix; @show suminter; @show gpnum; @show rmfl;

################################################################################

odesys, sdesys, fixparm, trueparents, prmrng, prmrep = datasettings(srcset, interclass, usefix)

x,y = simulate(odesys, sdesys, numspecies, tspan, δt, σ)

xnew, xmu, xvar, xdotmu, xdotvar = interpolate(x, y, rmfl, gpnum)

parsets = construct_parsets(numspecies, maxinter, fixparm, interclass)

optimise_models!(parsets, fixparm, xmu, xdotmu, interclass, prmrng, prmrep)

weight_models!(parsets)

edgeweights = weight_edges(parsets, suminter, interclass)

ranks = get_true_ranks(trueparents, parsets, suminter)

bestmodels = get_best_id(parsets, suminter)

truedges, othersum = edgesummary(edgeweights,trueparents)

truth, scrs = metricdata(edgeweights,trueparents)

println(average_precision_score(truth, scrs[:,1]))
println(roc_auc_score(truth, scrs[:,1]))
prcurve = precision_recall_curve(truth, scrs[:,1])[1:2]
roccurve = roc_curve(truth, scrs[:,1])[1:2]
PyPlot.plot(prcurve[2],prcurve[1])
PyPlot.plot(roccurve[1],roccurve[2])

networkinference(y)
