import ScikitLearn, DataFrames, Combinatorics
ScikitLearn.@sk_import metrics: (average_precision_score, precision_recall_curve,
									roc_auc_score, roc_curve)

struct TParset
	speciesnum::Int
	intercount::Int
	parents::Array{Int, 1}
	intertype::Array{Symbol,1}
end

# for filename in ["../logs/log.csv","../logs/log_gnw1.csv","../logs/log_gnw1_old.csv","../logs/log_gnw2.csv","../logs/logmean.csv","../logs/logmean_gnw1.csv","../logs/logmean_gnw1_old.csv","../logs/logmean_gnw2.csv"]
for filename in ["../logs/log_gnw1_1.csv","../logs/logmean_gnw1_1.csv"]
	datf = DataFrames.readtable(filename, separator='	')
	datf[:aupr_undir] = [0.0 for i = 1: size(datf,1)]
	datf[:aupr_dir] = [0.0 for i = 1: size(datf,1)]
	# println(datf)
	# error()

	for i in 1:size(datf,1)
		if datf[:interclass][i] == "nothing"
			if datf[:srcset][i] == "osc"
				number_of_genes = 5
				trueparents = [TParset(1, 2, [1, 5], [:Activation]),
								TParset(2, 2, [2, 1], [:Activation]),
								TParset(3, 2, [3, 1], [:Activation]),
								TParset(4, 3, [4, 1, 3], [:Activation, :Repression]),
								TParset(5, 3, [5, 2, 4], [:Repression, :Activation])]
			elseif datf[:srcset][i] == "lin"
				number_of_genes = 5
				trueparents = [TParset(1, 2, [1, 5], [:Activation]),
								TParset(2, 3, [2, 1, 5], [:Activation, :Repression]),
								TParset(3, 2, [3, 1], [:Activation]),
								TParset(4, 3, [4, 1, 3], [:Activation, :Repression]),
								TParset(5, 3, [5, 2, 4], [:Repression, :Activation])]
			elseif datf[:srcset][i] == "gnw"
				number_of_genes = 10
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
			elseif datf[:srcset][i] == "gnw2"
				number_of_genes = 10
				trueparents = [TParset(1, 5, [1,2,3,4,6], [:Repression,:Activation,:Repression,:Activation]),
								TParset(2, 5, [2,5,3,8,10], [:Repression, :Activation, :Activation, :Repression]),
								TParset(3, 6, [3,6,7], [:Repression,:Activation]),
								TParset(4, 7, [4,3,5,7,8,10], [:Activation,:Repression,:Repression,:Activation,:Activation]),
								TParset(5, 3, [5,6,10], [:Activation,:Repression]),
								TParset(6, 2, [6,7], [:Activation]),
								TParset(7, 1, [7], []),
								TParset(8, 3, [8,6,10], [:Activation,:Activation]),
								TParset(9, 4, [9,3,4,6], [:Activation,:Repression,:Repression]),
								TParset(10, 3, [10,6,7], [:Activation,:Repression])]
			end

			edgeweights = eval(parse(datf[:edgeweights][i]))

			truth = Vector{Bool}(0)
			score = Vector{Float64}(0)

			for edge in Combinatorics.combinations(collect(1:number_of_genes),2)
				inthere = false
				tempscore = 0.0
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

				for j in 1:size(edgeweights,1)
					if (edge[1] == edgeweights[j,1] && edge[2] == edgeweights[j,2]) || (edge[2] == edgeweights[j,1] && edge[1] in edgeweights[j,2])
						tempscore += edgeweights[j,4]
					end
				end
				push!(score,tempscore)
			end

			aupr_undir = average_precision_score(truth, score)
			datf[:aupr_undir][i] = aupr_undir
			datf[:aupr_dir][i] = datf[:aupr_bic][i]
		else
			if datf[:srcset][i] == "osc"
				number_of_genes = 5
				trueparents = [TParset(1, 1, [5], [:Activation]),
								TParset(2, 1, [1], [:Activation]),
								TParset(3, 1, [1], [:Activation]),
								TParset(4, 2, [1, 3], [:Activation, :Repression]),
								TParset(5, 2, [2, 4], [:Repression, :Activation])]
			elseif datf[:srcset][i] == "lin"
				number_of_genes = 5
				trueparents = [TParset(1, 1, [5], [:Activation]),
								TParset(2, 2, [1, 5], [:Activation, :Repression]),
								TParset(3, 1, [1], [:Activation]),
								TParset(4, 2, [1, 3], [:Activation, :Repression]),
								TParset(5, 2, [2, 4], [:Repression, :Activation])]
			elseif datf[:srcset][i] == "gnw"
				number_of_genes = 10
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
			elseif datf[:srcset][i] == "gnw2"
				number_of_genes = 10
				trueparents = [TParset(1, 4, [2,3,4,6], [:Repression,:Activation,:Repression,:Activation]),
								TParset(2, 4, [5,3,8,10], [:Repression, :Activation, :Activation, :Repression]),
								TParset(3, 2, [6,7], [:Repression,:Activation]),
								TParset(4, 5, [3,5,7,8,10], [:Activation,:Repression,:Repression,:Activation,:Activation]),
								TParset(5, 2, [6,10], [:Activation,:Repression]),
								TParset(6, 1, [7], [:Activation]),
								TParset(7, 0, [], []),
								TParset(8, 2, [6,10], [:Activation,:Activation]),
								TParset(9, 3, [3,4,6], [:Activation,:Repression,:Repression]),
								TParset(10, 2, [6,7], [:Activation,:Repression])]
			end

			edgeweights = eval(parse(datf[:edgeweights][i]))

			truth = Vector{Bool}(0)
			score = Vector{Float64}(0)

			for edge in Combinatorics.combinations(collect(1:number_of_genes),2)
				inthere = false
				tempscore = 0.0
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

				for j in 1:size(edgeweights,1)
					if (edge[1] == edgeweights[j,1] && edge[2] == edgeweights[j,2]) || (edge[2] == edgeweights[j,1] && edge[1] in edgeweights[j,2])
						tempscore += edgeweights[j,5]
					end
				end
				push!(score,tempscore)
			end

			aupr_undir = average_precision_score(truth, score)
			datf[:aupr_undir][i] = aupr_undir





			truth = Vector{Bool}(0)
			score = Vector{Float64}(0)
			for edge in Combinatorics.permutations(collect(1:number_of_genes),2)
				inthere = false
				tempscore = 0.0
				for parent in trueparents
					if edge[1] == parent.speciesnum && edge[2] in parent.parents
						inthere = true
					end
				end
				if inthere
					push!(truth, true)
				else
					push!(truth, false)
				end

				for j in 1:size(edgeweights,1)
					if edge[1] == edgeweights[j,1] && edge[2] == edgeweights[j,2]
						tempscore += edgeweights[j,5]
					end
				end
				push!(score,tempscore)
			end
			datf[:aupr_dir][i] = average_precision_score(truth, score)
		end
	end

	DataFrames.writetable(filename,datf,separator='	')
	println(filename)
end
