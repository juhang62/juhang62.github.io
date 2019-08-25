---
layout: post
title: test math 2
---

{% include mathjax.html %}

A Gaussian process is an infinite collection of random variables,
of which any finite subset of them form a multivariate normal distribution.
It is sorely determined by its mean and covariance function and is
often denoted as $$f(\cdot)\sim\mathcal{GP}(m(\cdot),k(\cdot,\cdot))$$,
where $m(x)$ is the mean function and $$k(x,x')$$ is the covariance
function. By definition, for a finite collection $$\{x_{1},\cdots,x_{n}\}$$,
$$[f(x_{1}),\cdots,f(x_{n})]^{T}$$ is a multivariate normal distribution
with mean $$m(x_{1}),\cdots,m(x_{n})]^{T}$$ and covariance $$cov(f(x_{i}),f(x_{j}))=k(x_{i},x_{j}).$$
This joint multivariate normal makes Gaussian processes desirable
for regression, in which inferences on new test points can be easily
made by conditioning on training data. Next we show how GP regression
is connected to linear regression. 
