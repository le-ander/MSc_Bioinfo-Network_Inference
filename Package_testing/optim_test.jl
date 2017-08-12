using Optim


f(p) = sum(0.1-0.4*xmu[:,1] + p[1]*((xmu[:,4] .^ p[2])./(p[3]^p[2] + xmu[:,4] .^ p[2])) - xdotmu[:,1])^2.0


lower = [0.0, 0.1, 0.0]
upper = [5.0, 5.0, 4.0]
initial_x = [1.0, 2.0, 1.0]

results = optimize(f, initial_x, lower, upper, Fminbox{NelderMead}())

results.minimum

f([2,2,1.5])
