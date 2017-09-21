# Gaussian Processes for Network Inference
This Repository contains the code for the final research project completed as part of the MSc in Bioinformatics and Theoretical Systems Biology at Imperial College London 2016/2017.

## Abstract
*Aims:* In this work we aim to devise and compare several related network inference approaches for gene regulatory networks. We thereby aim to identify solutions to problems encountered in previous studies of this field.

*Methods:* Our approaches are based on gradient matching for triaging putative models. We use Gaussian process regression to interpolate the data and obtain the associated gradients. From those, we evaluate the fits of parametric as well non-parametric candidate models to the data under various settings. We weight each model using the Bayesian Information Criterion in order to obtain a reconstructed network with weighted edges.

*Results:* We achieve an area under the precision-recall curve (AUPR) of up to 0.94 - 1.0 when inferring undirected interactions in a five-gene noise-free network, depending on the method employed. We are furthermore able to infer directional mechanistic information on interactions when enough information about the network is known (AUPR up to 0.94). We observe a significant drop in performance for realistically simulated ten-gene *in silico* gene expression data (AUPR up to 0.46 for undirected interactions and up to 0.18 for directed interaction type inference).

*Conclusion:* Our methods outperform recently developed information theoretical network inference approaches under certain conditions. Moreover, we identify possible causes for the decreased performance and unexpected trends seen for stochastic data. We see great potential for combining different approaches discussed in this work in order to obtain both high inference performance and detailed mechanistic information.

## Main Files Explained
**GPinf.jl** - Julia module containing all main functions used in this projects.

**call.jl** - Julia script used to generate results data for this project. Calls functions from GPinf.jl.

**call5x.jl** - As above, but generated data by averaging over 5 repeats of the same methods.
