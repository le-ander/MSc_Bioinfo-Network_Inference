import DifferentialEquations, Plots


function odesys(t,u,du) # Define ODE system
	du[1] = 0.1-0.4*u[1]+2.0*((u[5]^2.0)/(1.5^2.0+u[5]^2.0))
	du[2] = 0.2-0.4*u[2]+1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[5]/2.0)^1.0))
	du[3] = 0.2-0.4*u[3]+2.0*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))
	du[4] = 0.4-0.1*u[4]+1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[3]/1.0)^2.0))
	du[5] = 0.3-0.3*u[5]+2.0*((u[4]^2.0)/(1.0^2.0+u[4]^2.0))*(1.0/(1.0+(u[2]/0.5)^3.0))
end

function odesys2(t,u,du) # Define ODE system
	du[1] = 0.1-0.4*u[1]+2.0*((u[5]^2.0)/(1.5^2.0+u[5]^2.0))
	du[2] = 0.2-0.4*u[2]+1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[5]/2.0)^1.0))
	du[3] = 0.2-0.4*u[3]+2.0*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))
	du[4] = 0.4-0.1*u[4]+1.227242256857199*(1.0/(1.0+(u[2]/1.4045386059000364)^2.6966380392989544))*((u[3]^2.180893967023851)/(0.5656564474387958^2.180893967023851+u[3]^2.180893967023851))
	du[5] = 0.3-0.3*u[5]+2.0*((u[4]^2.0)/(1.0^2.0+u[4]^2.0))*(1.0/(1.0+(u[2]/0.5)^3.0))
end

function odesys3(t,u,du) # Define ODE system
	du[1] = 0.1-0.4*u[1]+3.818613140758792*((u[5]^1.5477088925196676)/(2.8894953037317372^1.5477088925196676+u[5]^1.5477088925196676))
	du[2] = 0.2-0.4*u[2]+0.5952550843766251*(((u[1]^3.052524204223092)/(3.197723576986178^3.052524204223092+u[1]^3.052524204223092))+((u[2]^1.405947950914821)/(0.5962085972540198^1.405947950914821+u[2]^1.405947950914821)))
	du[3] = 0.2-0.4*u[3]+0.805566440662191*(((u[1]^4.59416127584625)/(1.8410214321477527^4.59416127584625+u[1]^4.59416127584625)) + ((u[3]^4.467523511118766)/(0.9448670239269943^4.467523511118766+u[3]^4.467523511118766)))
	du[4] = 0.4-0.1*u[4]+1.227242256857199*(1.0/(1.0+(u[2]/1.4045386059000364)^2.6966380392989544))*((u[3]^2.180893967023851)/(0.5656564474387958^2.180893967023851+u[3]^2.180893967023851))
	du[5] = 0.3-0.3*u[5]+0.9172166580287355*(1.0/(1.0+(u[2]/0.8189618357198283)^3.286844146061471))*(1.0/(1.0+(u[5]/1.5622974931950642)^1.5502359361284548))
end


u0 = [1.0;0.5;1.0;1.5;0.5] # Define initial conditions
tspan = (0.0,20.0) # Define timespan for solving ODEs

prob = DifferentialEquations.ODEProblem(odesys,u0,tspan) # Formalise ODE problem
prob2 = DifferentialEquations.ODEProblem(odesys2,u0,tspan) # Formalise ODE problem
prob3 = DifferentialEquations.ODEProblem(odesys3,u0,tspan) # Formalise ODE problem

sol = DifferentialEquations.solve(prob, DifferentialEquations.RK4(), dt=1.0) # Solve ODEs with RK4 solver
sol2 = DifferentialEquations.solve(prob2, DifferentialEquations.RK4(), dt=1.0) # Solve ODEs with RK4 solver
sol3 = DifferentialEquations.solve(prob3, DifferentialEquations.RK4(), dt=1.0) # Solve ODEs with RK4 solver

Plots.plot(sol)
Plots.plot(sol2)
Plots.plot(sol3)
