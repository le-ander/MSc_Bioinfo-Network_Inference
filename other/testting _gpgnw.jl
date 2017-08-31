import Plots, PyCall

PyCall.@pyimport GPy.kern as gkern
PyCall.@pyimport GPy.models as gmodels
PyCall.@pyimport GPy.util.multioutput as gmulti

kernel = gkern.RBF(input_dim=1)
xmu = Array{Float64}(size(y))
xvar = Array{Float64}(size(y))
xdotmu = Array{Float64}(size(y))
xdotvar = Array{Float64}(size(y))

for i = 1:size(y,2)
	Y = reshape(y[:,i],(:,1))
	println(Y)
	m = gmodels.GPRegression(x,Y,kernel)

	# m[:rbf]["variance"][:constrain_fixed](1e-1)
	m[:rbf]["lengthscale"][:constrain_fixed](200.0)
	# m[:Gaussian_noise]["variance"][:constrain_fixed](0.1)

	m[:optimize_restarts](num_restarts = 5, verbose=false, parallel=true)
	#m[:plot](plot_density=false)

	# println(m[:param_array])

	vals = m[:predict](x.+1)
	deriv = m[:predict_jacobian](x)

	xmu[:,i] = vals[1][:]
	# xvar[:,i] = vals[2][:]
	# xdotmu[:,i] = deriv[1][:]
	# xdotvar[:,i] = deriv[2][:]
end

xnew = collect(reshape(1:10:1000,(:,1)))

Y = reshape(y[:,1],(:,1))
println(Y)
m = gmodels.GPRegression(x,Y,kernel)

# m[:rbf]["variance"][:constrain_fixed](1e-1)
m[:rbf]["lengthscale"][:constrain_fixed](200.0)
# m[:Gaussian_noise]["variance"][:constrain_fixed](0.1)

m[:optimize_restarts](num_restarts = 5, verbose=true, parallel=true)
m[:plot](plot_density=false)

println(m[:param_array])

vals = m[:predict](xnew)
deriv = m[:predict_jacobian](xnew)

Plots.plot(xnew,vals)
