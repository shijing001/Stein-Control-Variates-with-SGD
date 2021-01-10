# Stein Control Variates with SGD

This repository presents python3 code for the paper entitled ``Scalable Control Variates for Monte Carlo Methods via Stochastic Optimization" by Shijing Si, Chris Oates, Andrew B. Duncan, Lawrence Carin, and François-Xavier Briol (https://arxiv.org/abs/2006.07487). Specifically, we provide the code for polynomial, kernel and neural networks control variates.

## requirements:
python3.5+ 

scipy==1.5.2 

numpy==1.19.2

torch==1.6.0

sklearn==0.23.2


## Toy examples
In order to make it easy for people to use it, we provide an example in the `demonstration--polynomial_integrand_experiments.ipynb` file, where the integrand is polynomials of the form $\Sum_{i=1}^{m}\Sum_{j=1}^{n}A_{i,j}x_{i}^{j}$; and the probability measure is normal distribution with zero mean and standard deviation $\sigma$. 
