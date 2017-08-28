using DataFrames, StatPlots, Plots

df = readtable("logmean.csv", separator='	')

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
templ = df[(df[:dt] .== 1.0), :][:aupr_bic]
tempr = df[(df[:dt] .== 0.5), :][:aupr_bic]
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
"""
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
name = "B so GP / mo GP"
templ = df[(df[:gpnum] .== "nothing"), :][:aupr_bic]
tempr = df[(df[:gpnum] .== "5"), :][:aupr_bic]
leftaupr = vcat(leftaupr, templ)
rightaupr = vcat(rightaupr, tempr)
push!(leftfeature,[name for i in 1:length(templ)]...)
push!(rightfeature,[name for i in 1:length(tempr)]...)

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


left = DataFrame()
left[:Feature] = leftfeature
left[:AUPR] = leftaupr
right = DataFrame()
right[:Feature] = rightfeature
right[:AUPR] = rightaupr

myPlot = violin(left,:Feature,:AUPR, side=:left, marker=(0.2,:blue,stroke(0)), legend=false, ylims=(0,1.0))
violin!(right,:Feature,:AUPR, side=:right, marker=(0.2,:red,stroke(0)))
