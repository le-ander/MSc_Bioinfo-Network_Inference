using PyCall
using PyPlot

@pyimport numpy as np
@pyimport numpy.random as nr
@pyimport GPy
@pyimport GPy.kern as gkern
@pyimport GPy.models as gmodels
@pyimport GPy.plotting as gplotting

X = nr.uniform(-3.,3.,(20,1))
Y = np.sin(X) + nr.randn(20,1)*0.05

kernel = gkern.RBF(input_dim=1, variance=1., lengthscale=1.)

m = gmodels.GPRegression(X,Y,kernel)

m[:optimize_restarts](num_restarts = 10)

m[:plot](plot_density=true)
