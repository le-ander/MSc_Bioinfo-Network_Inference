import DifferentialEquations, Plots

function odesys(t,u,du)
	du[1] = 0.1-0.4*u[1]+2.0*((u[5]^2.0)/(1.5^2.0+u[5]^2.0))
	du[2] = 0.2-0.4*u[2]+1.5*((u[1]^2)/(1.5^2+u[1]^2))*(1/(1+(u[5]/2)^1))
	du[3] = 0.2-0.4*u[3]+2*((u[1]^2)/(1.5^2+u[1]^2))
	du[4] = 0.4-0.1*u[4]+1.5*((u[1]^2)/(1.5^2+u[1]^2))*(1/(1+(u[3]/1)^2))
	du[5] = 0.3-0.3*u[5]+2*((u[4]^2)/(1^2+u[4]^2))*(1/(1+(u[2]/0.5)^3))
end

u0 = [1.0;0.5;1.0;1.5;0.5]
tspan = (0.0,20.0)
prob = DifferentialEquations.ODEProblem(odesys,u0,tspan)

sol = DifferentialEquations.solve(prob, DifferentialEquations.RK4(), dt=1.0)

Plots.plot(sol)
