import ScikitLearn, DataFrames
ScikitLearn.@sk_import metrics: (average_precision_score, precision_recall_curve,
									roc_auc_score, roc_curve)

struct TParset
	speciesnum::Int
	intercount::Int
	parents::Array{Int, 1}
	intertype::Array{Symbol,1}
end

datf = DataFrames.readtable("logfull.csv", separator='	')

for i in 1:size(datf,1)
	if datf[:interclass][i] == "nothing"
		if datf[:srcset][i] == "osc"
			trueparents = [TParset(1, 2, [1, 5], [:Activation]),
							TParset(2, 2, [2, 1], [:Activation]),
							TParset(3, 2, [3, 1], [:Activation]),
							TParset(4, 3, [4, 1, 3], [:Activation, :Repression]),
							TParset(5, 3, [5, 2, 4], [:Repression, :Activation])]
		else
			trueparents = [TParset(1, 2, [1, 5], [:Activation]),
							TParset(2, 3, [2, 1, 5], [:Activation, :Repression]),
							TParset(3, 2, [3, 1], [:Activation]),
							TParset(4, 3, [4, 1, 3], [:Activation, :Repression]),
							TParset(5, 3, [5, 2, 4], [:Repression, :Activation])]
		end

		edgeweights = eval(parse(datf[:edgeweights][i]))

		truth = Vector{Bool}(0)
		scoreaic = Vector{Float64}(0)
		scorebic = Vector{Float64}(0)

		for j in 1:size(edgeweights,1)
			if edgeweights[j,1] == edgeweights[j,2]
				continue
			end
			push!(scoreaic,edgeweights[j,3])
			push!(scorebic,edgeweights[j,4])
			if edgeweights[j,2] in trueparents[convert(Int,edgeweights[j,1])].parents
				push!(truth,true)
			else
				push!(truth,false)
			end
		end

		aupr_aic = average_precision_score(truth, scoreaic)
		aupr_bic = average_precision_score(truth, scorebic)
		auroc_aic = roc_auc_score(truth, scoreaic)
		auroc_bic = roc_auc_score(truth, scorebic)

		println(datf[:aupr_aic][i], "\t", aupr_aic,"\t", i)
		datf[:aupr_aic][i] = aupr_aic
		datf[:aupr_bic][i] = aupr_bic
		datf[:auroc_aic][i] = auroc_aic
		datf[:auroc_bic][i] = auroc_bic
	end
end

DataFrames.writetable("log_corrected.csv",datf,separator='	')
