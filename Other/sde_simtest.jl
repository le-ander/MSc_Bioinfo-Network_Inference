import DifferentialEquations, Plots


# Define source data
numspecies = 5
srcset = :lin # :lin :osc :gnw

# Simulate source data
tspan = (0.0,20.0)
δt = 0.01
σ = :sde # Std.dev for ode + obs. noise or :sde

function oscodesys(t,u,du)
	du[1] = 0.2 - 0.9*u[1] + 2.0*((u[5]^5.0)/(1.5^5.0+u[5]^5.0))
	du[2] = 0.2 - 0.9*u[2] + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0))
	du[3] = 0.2 - 0.7*u[3] + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0))
	du[4] = 0.2 - 1.5*u[4] + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0)) + 2.0*(1.0/(1.0+(u[3]/1.5)^5.0))
	du[5] = 0.2 - 1.5*u[5] + 2.0*((u[4]^5.0)/(1.5^5.0+u[4]^5.0)) + 2.0*(1.0/(1.0+(u[2]/1.5)^3.0))
end
function oscsdesys(t, u, du)
	du[1,1] = √(0.2 + 2.0*((u[5]^5.0)/(1.5^5.0+u[5]^5.0))); du[1,6] = -√(0.9*u[1]);
	du[2,2] = √(0.2 + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0))); du[2,7] = -√(0.9*u[2]);
	du[3,3] = √(0.2 + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0))); du[3,8] = -√(0.7*u[3]);
	du[4,4] = √(0.2 + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0)) + 2.0*(1.0/(1.0+(u[3]/1.5)^5.0))); du[4,9] = -√(1.5*u[4]);
	du[5,5] = √(0.2 + 2.0*((u[4]^5.0)/(1.5^5.0+u[4]^5.0)) + 2.0*(1.0/(1.0+(u[2]/1.5)^3.0))); du[5,10] = -√(1.5*u[5]);
end
function linodesys(t,u,du)
	du[1] = 0.1-0.4*u[1]+2.0*((u[5]^2.0)/(1.5^2.0+u[5]^2.0))
	du[2] = 0.2-0.4*u[2]+1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[5]/2.0)^1.0))
	du[3] = 0.2-0.4*u[3]+2.0*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))
	du[4] = 0.4-0.1*u[4]+1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[3]/1.0)^2.0))
	du[5] = 0.3-0.3*u[5]+2.0*((u[4]^2.0)/(1.0^2.0+u[4]^2.0))*(1.0/(1.0+(u[2]/0.5)^3.0))
end
function linsdesys(t, u, du)
	du[1,1] =0.5 * √(0.1 + 2.0*((u[5]^2.0)/(1.5^2.0+u[5]^2.0))); du[1,6] = 0.5 * -√(0.4*u[1]);
	du[2,2] =0.5 * √(0.2 + 1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[5]/2.0)^1.0))); du[2,7] = 0.5 * -√(0.4*u[2]);
	du[3,3] =0.5 * √(0.2 + 2.0*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))); du[3,8] = 0.5 * -√(0.4*u[3]);
	du[4,4] =0.5 * √(0.4 + 1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[3]/1.0)^2.0))); du[4,9] = 0.5 * -√(0.1*u[4]);
	du[5,5] =0.5 * √(0.3 + 2.0*((u[4]^2.0)/(1.0^2.0+u[4]^2.0))*(1.0/(1.0+(u[2]/0.5)^3.0))); du[5,10] = 0.5 * -√(0.3*u[5]);
end


function simulateode(odesys, numspecies, tspan, step)
	u0 = [1.0;0.5;1.0;0.5;0.5] # Define initial conditions
	prob = DifferentialEquations.ODEProblem(odesys,u0,tspan) # Formalise ODE problem

	sol = DifferentialEquations.solve(prob, DifferentialEquations.RK4(), saveat=step) # Solve ODEs with RK4 solver
	x = reshape(sol.t,(length(sol.t),1))
	y = hcat(sol.u...)'
	sol
end

function simulatesde(odesys, sdesys, numspecies, tspan, step)
	A = hcat(eye(numspecies),-eye(numspecies))
	sparse(A)

	u0 = [1.0;0.5;1.0;0.5;0.5] # Define initial conditions

	prob = DifferentialEquations.SDEProblem(odesys,sdesys,u0,tspan,noise_rate_prototype=A)
	sol = DifferentialEquations.solve(prob, dt=0.001, saveat=step)
	x = reshape(sol.t,(length(sol.t),1))
	y = hcat(sol.u...)'
	sol
end

odesol = simulateode(linodesys, numspecies, tspan, δt)

sdesol = simulatesde(linodesys, linsdesys, numspecies, tspan, δt)

Plots.plot(odesol)

Plots.plot(sdesol)
