import DifferentialEquations, Plots

function oscgillespie(uzero, tstp)
	prob = DifferentialEquations.DiscreteProblem(uzero, tstp)
	# prob = DiscreteProblem([1.0;0.5;1.0;0.5;0.5],(0.0,20.0))

	rate = (t,u) -> 0.2 + 2.0*((u[5]^5.0)/(1.5^5.0+u[5]^5.0))
	affect! = function (integrator)
		integrator.u[1] += 1
	end
	jump1 = DifferentialEquations.ConstantRateJump(rate,affect!)

	rate = (t,u) -> 0.9*u[1]
	affect! = function (integrator)
		integrator.u[1] -= 1
	end
	jump2 = DifferentialEquations.ConstantRateJump(rate,affect!)


	rate = (t,u) -> 0.2 + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0))
	affect! = function (integrator)
		integrator.u[2] += 1
	end
	jump3 = DifferentialEquations.ConstantRateJump(rate,affect!)

	rate = (t,u) -> 0.9*u[2]
	affect! = function (integrator)
		integrator.u[2] -= 1
	end
	jump4 = DifferentialEquations.ConstantRateJump(rate,affect!)


	rate = (t,u) -> 0.2 + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0))
	affect! = function (integrator)
		integrator.u[3] += 1
	end
	jump5 = DifferentialEquations.ConstantRateJump(rate,affect!)

	rate = (t,u) -> 0.7*u[3]
	affect! = function (integrator)
		integrator.u[3] -= 1
	end
	jump6 = DifferentialEquations.ConstantRateJump(rate,affect!)


	rate = (t,u) -> 0.2 + 2.0*((u[1]^5.0)/(1.5^5.0+u[1]^5.0)) + 2.0*(1.0/(1.0+(u[3]/1.5)^5.0))
	affect! = function (integrator)
		integrator.u[4] += 1
	end
	jump7 = DifferentialEquations.ConstantRateJump(rate,affect!)

	rate = (t,u) -> 1.5*u[4]
	affect! = function (integrator)
		integrator.u[4] -= 1
	end
	jump8 = DifferentialEquations.ConstantRateJump(rate,affect!)


	rate = (t,u) -> 0.2 + 2.0*((u[4]^5.0)/(1.5^5.0+u[4]^5.0)) + 2.0*(1.0/(1.0+(u[2]/1.5)^3.0))
	affect! = function (integrator)
		integrator.u[5] += 1
	end
	jump9 = DifferentialEquations.ConstantRateJump(rate,affect!)

	rate = (t,u) -> 1.5*u[5]
	affect! = function (integrator)
		integrator.u[5] -= 1
	end
	jump10 = DifferentialEquations.ConstantRateJump(rate,affect!)



	jump_prob = DifferentialEquations.JumpProblem(prob,DifferentialEquations.Direct(),jump1,jump2,jump3,jump4,jump5,jump6,jump7,jump8,jump9,jump10)
	sol = DifferentialEquations.solve(jump_prob,DifferentialEquations.Discrete())
end

function lingillespie(uzero, tstp)
	prob = DifferentialEquations.DiscreteProblem(uzero, tstp)
	# prob = DiscreteProblem([1.0;0.5;1.0;0.5;0.5],(0.0,20.0))

	rate = (t,u) -> 0.1 + 2.0*((u[5]^2.0)/(1.5^2.0+u[5]^2.0))
	affect! = function (integrator)
		integrator.u[1] += 1
	end
	jump1 = DifferentialEquations.ConstantRateJump(rate,affect!)

	rate = (t,u) -> 0.4*u[1]
	affect! = function (integrator)
		integrator.u[1] -= 1
	end
	jump2 = DifferentialEquations.ConstantRateJump(rate,affect!)


	rate = (t,u) -> 0.2 + 1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[5]/2.0)^1.0))
	affect! = function (integrator)
		integrator.u[2] += 1
	end
	jump3 = DifferentialEquations.ConstantRateJump(rate,affect!)

	rate = (t,u) -> 0.4*u[2]
	affect! = function (integrator)
		integrator.u[2] -= 1
	end
	jump4 = DifferentialEquations.ConstantRateJump(rate,affect!)


	rate = (t,u) -> 0.2 + 2.0*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))
	affect! = function (integrator)
		integrator.u[3] += 1
	end
	jump5 = DifferentialEquations.ConstantRateJump(rate,affect!)

	rate = (t,u) -> 0.4*u[3]
	affect! = function (integrator)
		integrator.u[3] -= 1
	end
	jump6 = DifferentialEquations.ConstantRateJump(rate,affect!)


	rate = (t,u) -> 0.4 + 1.5*((u[1]^2.0)/(1.5^2.0+u[1]^2.0))*(1.0/(1.0+(u[3]/1.0)^2.0))
	affect! = function (integrator)
		integrator.u[4] += 1
	end
	jump7 = DifferentialEquations.ConstantRateJump(rate,affect!)

	rate = (t,u) -> 0.1*u[4]
	affect! = function (integrator)
		integrator.u[4] -= 1
	end
	jump8 = DifferentialEquations.ConstantRateJump(rate,affect!)


	rate = (t,u) -> 0.3 + 2.0*((u[4]^2.0)/(1.0^2.0+u[4]^2.0))*(1.0/(1.0+(u[2]/0.5)^3.0))
	affect! = function (integrator)
		integrator.u[5] += 1
	end
	jump9 = DifferentialEquations.ConstantRateJump(rate,affect!)

	rate = (t,u) -> 0.3*u[5]
	affect! = function (integrator)
		integrator.u[5] -= 1
	end
	jump10 = DifferentialEquations.ConstantRateJump(rate,affect!)



	jump_prob = DifferentialEquations.JumpProblem(prob,DifferentialEquations.Direct(),jump1,jump2,jump3,jump4,jump5,jump6,jump7,jump8,jump9,jump10)
	sol = DifferentialEquations.solve(jump_prob,DifferentialEquations.Discrete())
end

oscgsol = oscgillespie([1,1,1,1,1], (0.0,20.0))
lingsol = oscgillespie([1,1,1,1,1], (0.0,20.0))

Plots.plot(oscgsol)
Plots.plot(lingsol)
