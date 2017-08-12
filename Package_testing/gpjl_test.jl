using PyPlot, GaussianProcesses, Optim

# Training data
n = 10
x = 2π * rand(n)

y = sin(x) + 0.05*randn(n)


# Select mean and covariance function
mZero = MeanZero()                   # Zero mean function
kern = SE(0.0,0.0)                   # Squared exponential kernel with parameters
                                     # log(ℓ) = 0.0, log(σ) = 0.0


logObsNoise = -1.0
println(typeof(x))
x = reshape(x, (1,length(x)))                      # log standard deviation of observation noise (this is optional)
gp = GP(x,y,mZero,kern, logObsNoise)      # Fit the GP

μ, σ² = predict(gp,linspace(0,2π,100))

plot(gp)

optimize!(gp)

#######################################################################################

'''
using GaussianProcesses, PyPlot
nobsv=3000
X = randn(2,nobsv)
Y = randn(nobsv)
se = SEIso(0.0, 0.0)
gp = GP(X, Y, MeanConst(0.0), se, 0.0)
buf1=Array(Float64, nobsv,nobsv)
buf2=Array(Float64, nobsv,nobsv)
update_mll_and_dmll!(gp, buf1, buf2) # warm-up
@time update_mll_and_dmll!(gp, buf1, buf2)

plot(gp)
'''

########################################################################################



using Gadfly, GaussianProcesses

srand(13579)
# Training data
n=10;                          #number of training points
x = 2π * rand(n);              #predictors
y = sin.(x) + 0.05*randn(n);    #regressors
x = reshape(x, (1,length(x)))


#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

logObsNoise = -1.0                        # log standard deviation of observation noise (this is optional)
gp = GP(x,y,mZero,kern,logObsNoise)       #Fit the GP

optimize!(gp; method=Optim.BFGS())
