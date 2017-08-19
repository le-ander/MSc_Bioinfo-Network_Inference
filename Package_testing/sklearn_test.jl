using ScikitLearn, PyPlot
@sk_import metrics: (average_precision_score, precision_recall_curve)

average_precision_score([true, false, true, true],[0.9,0.1,0.5,0.05])

x,y,thresh = precision_recall_curve([1,0,1,1],[0.9,0.1,0.5,0.05])

Plots.plot(x,y)
