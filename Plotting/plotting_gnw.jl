using DataFrames, StatPlots, Plots
Plots.pyplot()

df = readtable("../logs/logmean_gnw1_1.csv", separator='	')

leftfeature = Vector{String}(0)
rightfeature = Vector{String}(0)

name = "A Osc. / Non-osc."
templ = df[(df[:srcset] .== "osc"), :][:aupr_bic]
tempr = df[(df[:srcset] .== "lin"), :][:aupr_bic]
leftaupr = templ
rightaupr = tempr
push!(leftfeature,[name for i in 1:length(templ)]...)
push!(rightfeature,[name for i in 1:length(tempr)]...)

name = "C 21 / 41"
templ = df[(df[:dt] .== 50.0), :][:aupr_bic]
tempr = df[(df[:dt] .== 25.0), :][:aupr_bic]
leftaupr = vcat(leftaupr, templ)
rightaupr = vcat(rightaupr, tempr)
push!(leftfeature,[name for i in 1:length(templ)]...)
push!(rightfeature,[name for i in 1:length(tempr)]...)

name = "D 2 par. / 3 par."
templ = df[(df[:maxinter] .== 2), :][:aupr_bic]
tempr = df[(df[:maxinter] .== 3), :][:aupr_bic]
leftaupr = vcat(leftaupr, templ)
rightaupr = vcat(rightaupr, tempr)
push!(leftfeature,[name for i in 1:length(templ)]...)
push!(rightfeature,[name for i in 1:length(tempr)]...)

name = "B so GP / mo GP"
templ = df[(df[:gpnum] .== "nothing"), :][:aupr_bic]
tempr = df[(df[:gpnum] .== "10"), :][:aupr_bic]
leftaupr = vcat(leftaupr, templ)
rightaupr = vcat(rightaupr, tempr)
push!(leftfeature,[name for i in 1:length(templ)]...)
push!(rightfeature,[name for i in 1:length(tempr)]...)

name = "E 50 / 100"
templ = df[(df[:lengthscale] .== 50.0), :][:aupr_bic]
tempr = df[(df[:lengthscale] .== 100.0), :][:aupr_bic]
leftaupr = vcat(leftaupr, templ)
rightaupr = vcat(rightaupr, tempr)
push!(leftfeature,[name for i in 1:length(templ)]...)
push!(rightfeature,[name for i in 1:length(tempr)]...)

name = "F 150 / 200"
templ = df[(df[:lengthscale] .== 150.0), :][:aupr_bic]
tempr = df[(df[:lengthscale] .== 200.0), :][:aupr_bic]
leftaupr = vcat(leftaupr, templ)
rightaupr = vcat(rightaupr, tempr)
push!(leftfeature,[name for i in 1:length(templ)]...)
push!(rightfeature,[name for i in 1:length(tempr)]...)

"""
name = "H GP / PIDC"
templ = df[(df[:interclass] .== "nothing"), :][:aupr_bic]
tempr = df[:thalia_aupr]
leftaupr = vcat(leftaupr, templ)
rightaupr = vcat(rightaupr, tempr)
push!(leftfeature,[name for i in 1:length(templ)]...)
push!(rightfeature,[name for i in 1:length(tempr)]...)

name = "I ODE f / PIDC"
templ = df[(df[:interclass] .!= "nothing") & (df[:usefix] .== true), :][:aupr_bic]
tempr = df[:thalia_aupr]
leftaupr = vcat(leftaupr, templ)
rightaupr = vcat(rightaupr, tempr)
push!(leftfeature,[name for i in 1:length(templ)]...)
push!(rightfeature,[name for i in 1:length(tempr)]...)

name = "J ODE nf / PIDC"
templ = df[(df[:interclass] .!= "nothing") & (df[:usefix] .== false), :][:aupr_bic]
tempr = df[:thalia_aupr]
leftaupr = vcat(leftaupr, templ)
rightaupr = vcat(rightaupr, tempr)
push!(leftfeature,[name for i in 1:length(templ)]...)
push!(rightfeature,[name for i in 1:length(tempr)]...)


name = "F ODE f / GP"
templ = df[(df[:interclass] .!= "nothing") & (df[:usefix] .== true), :][:aupr_bic]
tempr = df[(df[:interclass] .== "nothing"), :][:aupr_bic]
leftaupr = vcat(leftaupr, templ)
rightaupr = vcat(rightaupr, tempr)
push!(leftfeature,[name for i in 1:length(templ)]...)
push!(rightfeature,[name for i in 1:length(tempr)]...)

name = "G ODE nf / GP"
templ = df[(df[:interclass] .!= "nothing") & (df[:usefix] .== false), :][:aupr_bic]
tempr = df[(df[:interclass] .== "nothing"), :][:aupr_bic]
leftaupr = vcat(leftaupr, templ)
rightaupr = vcat(rightaupr, tempr)
push!(leftfeature,[name for i in 1:length(templ)]...)
push!(rightfeature,[name for i in 1:length(tempr)]...)

name = "E ODE f /ODE nf"
templ = df[(df[:interclass] .!= "nothing") & (df[:usefix] .== true), :][:aupr_bic]
tempr = df[(df[:interclass] .!= "nothing") & (df[:usefix] .== false), :][:aupr_bic]
leftaupr = vcat(leftaupr, templ)
rightaupr = vcat(rightaupr, tempr)
push!(leftfeature,[name for i in 1:length(templ)]...)
push!(rightfeature,[name for i in 1:length(tempr)]...)
"""


left = DataFrame()
left[:Feature] = leftfeature
left[:AUPR] = leftaupr
right = DataFrame()
right[:Feature] = rightfeature
right[:AUPR] = rightaupr

myPlot = violin(left,:Feature,:AUPR, side=:left, marker=(0.2), legend=false, ylims=(0,1.0), color=:black, xaxis=font(12), yaxis=(font(12)), line=[nothing, nothing, nothing, nothing, nothing])
violin!(right,:Feature,:AUPR, side=:right, marker=(0.2), color=:grey, line=[nothing, nothing, nothing, nothing, nothing])

# gui()
