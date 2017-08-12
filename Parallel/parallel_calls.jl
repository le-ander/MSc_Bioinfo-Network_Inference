addprocs(7)
@everywhere include("parallel_functions.jl")

x, y = simulate()


const numspecies = size(y,2)
const maxinter = 2
const interactions = [:Activation, :Repression]
const fixparm = [0.1 0.2 0.2 0.4 0.3; 0.4 0.4 0.4 0.1 0.3]

xmu, xvar, xdotmu, xdotvar = interpolate(x, y)

parsets = construct_parsets(numspecies, maxinter, interactions, selfinter=false)

optimise_models!(parsets, fixparm, xmu, xdotmu)

weight_models!(parsets)

edgeweights = weight_edges(parsets, interactions)
