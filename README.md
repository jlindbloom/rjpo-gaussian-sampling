# rjpo-gaussian-sampling

This repository provides a Python implementation of the reversible jump perturbation optimization (RJPO) method presented in [[1]](#1) for sampling high-dimensional Gaussians. This method avoids matrix factorizations and requires only matrix-vector products related to the precision matrix. The samples generated are inexact but valid in the MCMC sense.

In this implementation, we assume that our goal is to sample from the Gaussian $\mathcal{N}(\mu, Q^{-1})$, with $Q \in \mathbb{R}^n$ a SPD precision matrix. We assume the precision matrix is of the form
```math
Q = \sum_{i=1}^K L_i^T L_i,
```
where the user provides ``LinearOperator``s defining the $L_i$. The $L_i$ need not be square.



## References
<a id="1">[1]</a> 
C. Gilavert, S. Moussaoui and J. Idier, "Efficient Gaussian Sampling for Solving Large-Scale Inverse Problems Using MCMC," in IEEE Transactions on Signal Processing, vol. 63, no. 1, pp. 70-80, Jan.1, 2015, doi: 10.1109/TSP.2014.2367457.
