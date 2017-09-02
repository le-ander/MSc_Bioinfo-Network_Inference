push!(LOAD_PATH, "/cluster/home/ld2113/work/Final-Project/")
push!(LOAD_PATH, "/media/leander/Daten/Data/Imperial/Projects/Final Project/Final-Project")

using GPinf


# Define source data
numspecies = 10
srcset = :gnw # :lin :osc :gnw
gnwpath = "/cluster/home/ld2113/work/data/thalia-simulated/InSilicoSize10-Yeast1_dream4_timeseries_one.tsv"

# Simulate source data
tspan = (0.0,20.0)
δt = 50.0
σ = 0.0 # Std.dev for ode + obs. noise or :sde

# Define possible parent sets
maxinter = 2
interclass = nothing # :add :mult nothing
lengthscale= 100.0
usefix = false	#ODEonly

suminter = false	#ODEonly

gpnum = 10 # For multioutput gp: how many outputs at once, for single: nothing

rmfl = false

@show numspecies; @show srcset; @show tspan; @show δt; @show σ; @show maxinter;
@show interclass; @show lengthscale; @show usefix; @show suminter; @show gpnum; @show rmfl;

################################################################################

odesys, sdesys, fixparm, trueparents, prmrng, prmrep = datasettings(srcset, interclass, usefix)

if srcset == :gnw
	x, y = readgenes(gnwpath)
else
	x, y = simulate(odesys, sdesys, numspecies, tspan, δt, σ)
end

xnew, xmu, xvar, xdotmu, xdotvar = interpolate(x, y, δt, lengthscale, rmfl, gpnum)

parsets = construct_parsets(numspecies, maxinter, fixparm, interclass)

optimise_models!(parsets, fixparm, xmu, xdotmu, interclass, prmrng, prmrep)

weight_models!(parsets)

edgeweights = weight_edges(parsets, suminter, interclass)

ranks = get_true_ranks(trueparents, parsets, suminter)

bestmodels = get_best_id(parsets, suminter)

truedges, othersum = edgesummary(edgeweights,trueparents)

aupr_aic, aupr_bic, auroc_aic, auroc_bic = performance(edgeweights,trueparents)

println("AUPR AIC ", aupr_aic)
println("AUPR BIC ", aupr_bic)
println("AUROC AIC ", auroc_aic)
println("AUROC BIC ", auroc_bic)

# # output, thalia_aupr, thalia_auroc = networkinference(y, trueparents)
# println("PIDC AUPR ", thalia_aupr)
# println("PIDC AUROC ", thalia_auroc)
#
# cnt = float(readline("count.txt"))
# write("count.txt", string(cnt+1))
#
# dtime = now()
#
# open("../logs/log_gnw.csv", "a") do f
# 	# write(f, "id\ttime\tnumspecies\tsrcset\ttspan\tdt\tsigma\tmaxinter\tinterclass\tlengthscale\tusefix\tsuminter\tgpnum\trmfl\taupr_aic\taupr_bic\tauroc_aic\tauroc_bic\tthalia_aupr\tthalia_auroc\tranks\tbestmodels\ttruedges\tothersum\tedgeweights")
# 	# write(f, "\n")
# 	write(f, "$cnt\t$dtime\t$numspecies\t$srcset\t$tspan\t$δt\t$σ\t$maxinter\t$interclass\t$lengthscale\t$usefix\t$suminter\t$gpnum\t$rmfl\t$aupr_aic\t$aupr_bic\t$auroc_aic\t$auroc_bic\t$thalia_aupr\t$thalia_auroc\t$ranks\t$bestmodels\t$truedges\t$othersum\t$edgeweights")
# 	write(f, "\n")
# end
