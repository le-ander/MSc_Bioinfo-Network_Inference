using DataFrames, StatPlots, Plots
pyplot()

df = readtable("../logs/logmean.csv", separator='	')

feature = Vector{String}(0)

name = "1:ODE constr. (T)"
templ = df[(df[:interclass] .!= "nothing") & (df[:usefix] .== true), :][:aupr_bic]
aupr = templ
push!(feature,[name for i in 1:length(templ)]...)

name = "2:ODE unconstr. (T)"
templ = df[(df[:interclass] .!= "nothing") & (df[:usefix] .== false), :][:aupr_bic]
aupr = vcat(aupr, templ)
push!(feature,[name for i in 1:length(templ)]...)

################################################################################

name = "3:ODE constr. (D)"
templ = df[(df[:interclass] .!= "nothing") & (df[:usefix] .== true), :][:aupr_dir]
aupr = vcat(aupr, templ)
push!(feature,[name for i in 1:length(templ)]...)

name = "4:ODE unconstr. (D)"
templ = df[(df[:interclass] .!= "nothing") & (df[:usefix] .== false), :][:aupr_dir]
aupr = vcat(aupr, templ)
push!(feature,[name for i in 1:length(templ)]...)

name = "5:GP (D)"
templ = df[(df[:interclass] .== "nothing"), :][:aupr_bic]
aupr = vcat(aupr, templ)
push!(feature,[name for i in 1:length(templ)]...)

################################################################################

name = "6:ODE constr. (U)"
templ = df[(df[:interclass] .!= "nothing") & (df[:usefix] .== true), :][:aupr_undir]
aupr = vcat(aupr, templ)
push!(feature,[name for i in 1:length(templ)]...)

name = "7:ODE unconstr. (U)"
templ = df[(df[:interclass] .!= "nothing") & (df[:usefix] .== false), :][:aupr_undir]
aupr = vcat(aupr, templ)
push!(feature,[name for i in 1:length(templ)]...)

name = "8:GP (U)"
templ = df[(df[:interclass] .== "nothing"), :][:aupr_undir]
aupr = vcat(aupr, templ)
push!(feature,[name for i in 1:length(templ)]...)

name = "9:PIDC (U)"
templ = df[:thalia_aupr]
aupr = vcat(aupr, templ)
push!(feature,[name for i in 1:length(templ)]...)


left = DataFrame()
left[:Feature] = feature
left[:AUPR] = aupr

myPlot = violin(left,:Feature,:AUPR, marker=(0.2), legend=false, ylims=(0,1.0), color=[:black,:black,:grey,:grey,:grey,:lightgrey,:lightgrey,:lightgrey,:white], xlabel="Network Inference Method" , ylabel="Area under the Precision-Recall Curve",xaxis=font(12), yaxis=(font(12)), line=[nothing, nothing, nothing, nothing, nothing, :black, :black, :black, :black])

gui()
# savefig("test.pdf")
