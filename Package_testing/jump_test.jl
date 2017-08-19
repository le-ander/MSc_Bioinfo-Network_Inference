using JuMP
using NLopt

m = Model(solver = NLoptSolver(algorithm = :LD_MMA))
@variable(m, x, start = 0.0)
@variable(m, y, start = 0.0)
@constraint(m, x >= 2)

@NLobjective(m, Min, (1-x)^2 + 100(y-x^2)^2)

print(m)

solve(m)
println("x = ", getvalue(x), " y = ", getvalue(y))

# adding a (linear) constraint
@constraint(m, x + y = 10)
solve(m)
println("x = ", getvalue(x), " y = ", getvalue(y))





function optimise_params(xmu, xdotmu)
	m = JuMP.Model(solver = NLopt.NLoptSolver(algorithm = :LD_MMA))
	JuMP.@variable(m, 0.0 <= β <= 5.0 )
	JuMP.@variable(m, 0.0 <= θ <= 4.0 )
	JuMP.@variable(m, 0.1 <= ω <= 5.0 )

	JuMP.@NLobjective(m, Min, sum((0.1-0.4*xmu[:,1] + β*((xmu[:,5] .^ ω)./(θ^ω + xmu[:,5] .^ ω)) - xdotmu[:,1])^2.0))

	JuMP.solve(m)
	println("x = ", getvalue(x), " y = ", getvalue(y))
end
