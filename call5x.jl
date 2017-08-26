push!(LOAD_PATH, "/cluster/home/ld2113/work/Final-Project/")
push!(LOAD_PATH, "/media/leander/Daten/Data/Imperial/Projects/Final Project/Final-Project")

using GPinf


# Define source data
numspecies = 5
srcset = :osc # :lin :osc :gnw
gnwpath = "/cluster/home/ld2113/work/data/thalia-simulated/InSilicoSize10-Yeast1_dream4_timeseries_one.tsv"

# Simulate source data
tspan = (0.0,20.0)
δt = 0.5
σ = 0.0 # Std.dev for ode + obs. noise or :sde

# Define possible parent sets
maxinter = 2
interclass = :add # :add :mult nothing
usefix = false	#ODEonly

suminter = false	#ODEonly

gpnum = numspecies # For multioutput gp: how many outputs at once, for single: nothing

rmfl = false

@show numspecies; @show srcset; @show tspan; @show δt; @show σ; @show maxinter;
@show interclass; @show usefix; @show suminter; @show gpnum; @show rmfl;

################################################################################

repeats = 5

cnt = float(readline("count.txt"))
write("count.txt", string(cnt+5))

if interclass == nothing
	avgindx = 3
else
	avgindx = 4
end
avg_aupr_aic = 0.0
avg_aupr_bic = 0.0
avg_auroc_aic = 0.0
avg_auroc_bic = 0.0
avg_thalia_aupr = 0.0
avg_thalia_auroc = 0.0
avg_othersum = 0.0
avg_truedges = [0.0 0.0]
avg_edgeweights = [0.0 0.0]

for i = 1:repeats
	println("RUN NUMBER $i")

	odesys, sdesys, fixparm, trueparents, prmrng, prmrep = datasettings(srcset, interclass, usefix)

	if srcset == :gnw
		x, y = readgenes(gnwpath)
	else
		x, y = simulate(odesys, sdesys, numspecies, tspan, δt, σ)
	end

	xnew, xmu, xvar, xdotmu, xdotvar = interpolate(x, y, rmfl, gpnum)

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

	output, thalia_aupr, thalia_auroc = networkinference(y, trueparents)
	println("PIDC AUPR ", thalia_aupr)
	println("PIDC AUROC ", thalia_auroc)

	open("log.csv", "a") do f
		# write(f, "id\ttime\tnumspecies\tsrcset\ttspan\tdt\tsigma\tmaxinter\tinterclass\tusefix\tsuminter\tgpnum\trmfl\taupr_aic\taupr_bic\tauroc_aic\tauroc_bic\tthalia_aupr\tthalia_auroc\tranks\tbestmodels\ttruedges\tothersum\tedgeweights")
		# write(f, "\n")
		write(f, "$(cnt+i-1)\t$(now())\t$numspecies\t$srcset\t$tspan\t$δt\t$σ\t$maxinter\t$interclass\t$usefix\t$suminter\t$gpnum\t$rmfl\t$aupr_aic\t$aupr_bic\t$auroc_aic\t$auroc_bic\t$thalia_aupr\t$thalia_auroc\t$ranks\t$bestmodels\t$truedges\t$othersum\t$edgeweights")
		write(f, "\n")
	end

	avg_aupr_aic += aupr_aic / repeats
	avg_aupr_bic += aupr_bic / repeats
	avg_auroc_aic += auroc_aic / repeats
	avg_auroc_bic += auroc_bic / repeats
	avg_thalia_aupr += thalia_aupr / repeats
	avg_thalia_auroc += thalia_auroc / repeats
	avg_othersum += othersum / repeats

	if i == 1
		avg_truedges = truedges
		avg_edgeweights = edgeweights
	else
		avg_truedges[:,[avgindx, avgindx+1]] .+= truedges[:,[3,4]]
		avg_edgeweights[:,[avgindx, avgindx+1]] .+= edgeweights[:,[3,4]]
	end
end

avg_truedges[:,[avgindx, avgindx+1]] ./= repeats
avg_edgeweights[:,[avgindx, avgindx+1]] ./= repeats

open("logmean.csv", "a") do f
	# write(f, "id\ttime\tnumspecies\tsrcset\ttspan\tdt\tsigma\tmaxinter\tinterclass\tusefix\tsuminter\tgpnum\trmfl\taupr_aic\taupr_bic\tauroc_aic\tauroc_bic\tthalia_aupr\tthalia_auroc\ttruedges\tothersum\tedgeweights")
	# write(f, "\n")
	write(f, "$cnt\t$(now())\t$numspecies\t$srcset\t$tspan\t$δt\t$σ\t$maxinter\t$interclass\t$usefix\t$suminter\t$gpnum\t$rmfl\t$avg_aupr_aic\t$avg_aupr_bic\t$avg_auroc_aic\t$avg_auroc_bic\t$avg_thalia_aupr\t$avg_thalia_auroc\t$avg_truedges\t$avg_othersum\t$avg_edgeweights")
	write(f, "\n")
end
