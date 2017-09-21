import Plots, PyCall, PyPlot
Plots.pyplot()

PyCall.@pyimport GPy.kern as gkern
PyCall.@pyimport GPy.models as gmodels
PyCall.@pyimport GPy.util.multioutput as gmulti

kernel = gkern.RBF(input_dim=1)
xmu = Array{Float64}(size(y))
xvar = Array{Float64}(size(y))
xdotmu = Array{Float64}(size(y))
xdotvar = Array{Float64}(size(y))

# for i = 1:size(y,2)
# 	Y = reshape(y[:,i],(:,1))
# 	println(Y)
# 	m = gmodels.GPRegression(x,Y,kernel)
#
# 	# m[:rbf]["variance"][:constrain_fixed](1e-1)
# 	m[:rbf]["lengthscale"][:constrain_fixed](200.0)
# 	# m[:Gaussian_noise]["variance"][:constrain_fixed](0.1)
#
# 	m[:optimize_restarts](num_restarts = 5, verbose=false, parallel=true)
# 	#m[:plot](plot_density=false)
#
# 	# println(m[:param_array])
#
# 	vals = m[:predict](x.+1)
# 	deriv = m[:predict_jacobian](x)
#
# 	xmu[:,i] = vals[1][:]
# 	# xvar[:,i] = vals[2][:]
# 	# xdotmu[:,i] = deriv[1][:]
# 	# xdotvar[:,i] = deriv[2][:]
# end

# x = [n for n = 0:20][:,:]


xnew = collect(reshape(0:0.1:20,(:,1)))
# xnew = collect(reshape(0:10:1000,(:,1)))


Y = reshape(y[:,5],(:,1))
println(Y)
m = gmodels.GPRegression(x,Y,kernel)

m[:param_array]
# m[:rbf]["lengthscale"] = 150

# m[:rbf]["variance"][:constrain_fixed](1e-1)
# m[:rbf]["lengthscale"][:constrain_fixed](150)
# m[:rbf]["lengthscale"][:constrain_bounded](50,600)
# m[:Gaussian_noise]["variance"][:constrain_fixed](0.001)

m[:optimize_restarts](num_restarts = 5, verbose=true, parallel=true)
m[:plot](plot_density=false)

println(m[:param_array])

vals = m[:predict](xnew)
deriv = m[:predict_jacobian](xnew)
quantiles = m[:predict_quantiles](xnew)

# Plots.plot(xnew,vals)
Plots.plot(xnew,[vals[1] vals[1]], fillrange=[vals[1]-2*sqrt.(vals[2]) vals[1]+2*sqrt.(vals[2])], fillalpha=0.3, legend=false, color=:black, ylimit=(0,2.5))


################################################################################


ytemp = [reshape(y[:,i],(:,1)) for i = 1:5]
icm = gmulti.ICM(input_dim=1,num_outputs=5,kernel=gkern.RBF(1))
m = gmodels.GPCoregionalizedRegression([x for i in 1:5],ytemp,kernel=icm)

# m[:ICM][:rbf]["lengthscale"] = 150

# m[:ICM][:rbf]["variance"][:constrain_fixed](1e-1)
# m[:ICM][:rbf]["lengthscale"][:constrain_fixed](100)
# m[:mixed_noise][:constrain_fixed](0.1)

m[:optimize_restarts](num_restarts = 16, verbose=true, parallel=true)

println(m[:param_array])
println(m[:parameter_names]())

# xnew=[n for n = 0:10:1000]
xnew=[n for n = 0:0.1:20]
xmu = zeros(201,5)
xvar = zeros(201,5)
for species in 1:5
	prediction = m[:predict](hcat(xnew,[species-1 for t in xnew]), Y_metadata=Dict("output_index" => Int[species-1 for t in xnew]))
	quantiles = m[:predict_quantiles](hcat(xnew,[species-1 for t in xnew]), Y_metadata=Dict("output_index" => Int[species-1 for t in xnew]))
	xmu[:,species] .+= prediction[1][:]
	xvar[:,species] .+= prediction[2][:]
end

Plots.plot(xnew,[xmu[:,5] xmu[:,5]], fillrange=[xmu[:,5]-2*sqrt.(xvar[:,5]) xmu[:,5]+2*sqrt.(xvar[:,5])], fillalpha=0.3, color=:black, ylimit=(0,2.5), legend=false)
