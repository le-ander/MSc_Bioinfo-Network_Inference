using Plots
pyplot()

u0 = [1.0;0.5;1.0;0.5;0.5]

prob = DifferentialEquations.ODEProblem(odesys,u0,tspan)
sol = DifferentialEquations.solve(prob, DifferentialEquations.RK4())

plot(sol,ylims=(0.0,3.0),xlabel="Time",ylabel="Gene Expression", guidefont=font(16), tickfont=font(12), legendfont=font(8))

plot(sol,ylims=(0.0,5.0),xlabel="Time",ylabel="Gene Expression", guidefont=font(16), tickfont=font(12), legendfont=font(8))

plot(x,y, line=3.0,ylims=(0.0,1.0),xlabel="Time",ylabel="Gene Expression", guidefont=font(16), tickfont=font(12), legendfont=font(8), legend=:bottomright)

gui()
