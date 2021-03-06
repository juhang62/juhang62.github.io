---
layout: post
title: How A Gaussian Process Arises as a Limit of Linear Regression
---

{% include mathjax.html %}

Gaussian process (GP) models are very useful in machine learning tasks.
Not only it naturally offers probabilistic interpretation but also
shines in the tasks where data is not so abundant. It has connections
to many other machine learning methods such as support vector machine
and neural networks. In this post, we note that GP regression is linear
regression with Gaussian prior on parameters, and sketch how GP can
arise as a limit of linear model as the number of features goes to
infinite.

A Gaussian process is an infinite collection of random variables,
of which any finite subset of them form a multivariate normal distribution.
It is sorely determined by its mean and covariance function and is
often denoted as $f(\cdot)\sim\mathcal{GP}(m(\cdot),k(\cdot,\cdot))$,
where $m(x)$ is the mean function and $k(x,x')$ is the covariance
function. By definition, for a finite collection $\{x_{1},\cdots,x_{n}\}$,
$[f(x_{1}),\cdots,f(x_{n})]^{T}$ is a multivariate normal distribution
with mean $m(x_{1}),\cdots,m(x_{n})]^{T}$ and covariance $cov(f(x_{i}),f(x_{j}))=k(x_{i},x_{j}$).
This joint multivariate normal makes Gaussian processes desirable
for regression, in which inferences on new test points can be easily
made by conditioning on training data. Next we show how GP regression
is connected to linear regression. 

A linear regression model can be written as 
$$\begin{eqnarray}
f & = & \mathbf{x}^{T}\mathbf{w}\label{eq:f=xw}\\
y & = & f(\mathbf{x})+\epsilon\nonumber 
\end{eqnarray}$$
where $\mathbf{x}$ is the input vector, $\mathbf{w}$ is the parameter
vector, $\epsilon\sim\mathcal{N}(0,\sigma_{n}^{2}$) is noise. In
a Bayesian approach, we assume a prior $\mathbf{w}\sim\mathcal{N}(0,\sigma^{2}I)$.
Thus given certain $\mathbf{x}$, $f$ is a linear combination of
normal random variables and hence is also normal. Moreover, for a
finite collection of $\mathbf{x}$, their corresponding $f$'s is
a multivariate Gaussian. Therefore, $f$ is a Gaussian process. Its
mean function is

$$m(\boldsymbol{x})=E[f(\mathbf{x})]=E[\mathbf{x}^{T}\mathbf{w}]=0,$$

and its covariance function is 
$$\begin{eqnarray}
k(\mathbf{x},\mathbf{x}') & = & cov(f(\mathbf{x}),f(\mathbf{x}'))\nonumber \\
 & = & E[\mathbf{x}^{T}\mathbf{w}\mathbf{x}^{T}\mathbf{w}]\nonumber \\
 & = & \sigma^{2}\mathbf{x}^{T}\mathbf{x}'\label{eq:xTx}
\end{eqnarray}$$
which is called dot product covariance function.

The simple linear regression is restricted by the fact it only represents
linear relationship with $\mathbf{x}$. A common way to resolve with
this limitation is to expand the feature space by using basis functions
$\phi_{i}(x)$, so instead of \eqref{eq:f=xw}, we have $f=\boldsymbol{\phi}(\mathbf{x})^{T}\mathbf{w}$.
For illustrating purpose, we assume input vector $\mathbf{x}$ is
1-dimensional. Some examples of basis functions include monomials
$\boldsymbol{\phi}(x)=[1,x,x^{2},\cdots]$, Fourier $\boldsymbol{\phi}(x)=[\sin(x),\cos(x),\sin(2x),\cos(2x),\cdots]$,
radial basis functions $\boldsymbol{\phi}(x)=[e^{-(x-1)^{2}},e^{-(x-2)^{2}},e^{-(x-3)^{2}},\cdots]$.
Since the model remains linear in $\mathbf{w},$ results from simple
linear regression are extended directly. In particular,\eqref{eq:xTx}
becomes $k(\mathbf{x},\mathbf{x}')=\sigma^{2}\boldsymbol{\phi}(\mathbf{x})^{T}\boldsymbol{\phi}(\mathbf{x}')$.Considering a set of $N$ radial basis functions centered equidistantly,
it follows that 


$$\begin{equation}
k(\mathbf{x},\mathbf{x}')  =   \frac{\sigma^{2}}{N}\sum_{i=1}^{N}\exp(-\frac{(x-c_{i})^{2}}{2\ell^{2}})\exp(-\frac{(x'-c_{i})^{2}}{2\ell^{2}})
\end{equation}$$


where we have scaled the variance of $\mathbf{w}$ to be $\sigma^{2}/N$.
Formally, we let $N\to\infty$ and obtain


$$\begin{eqnarray}
k(\mathbf{x},\mathbf{x}') & \to & \sigma^{2}\int_{-\infty}^{\infty}\exp(-\frac{(x-c_{i})^{2}}{2\ell})\exp(-\frac{(x'-c_{i})^{2}}{2\ell})dc \nonumber  \\ 
 & = & \sqrt{\pi}\sigma^{2}\exp(-\frac{(x-x')^{2}}{2(\sqrt{2}\ell)^{2}}) \nonumber 
\end{eqnarray}$$


which is the widely used so-called squared exponential covariance
function. 

Building the connection to linear models not only offers another way
of viewing GP, but also opens doors to design and pick appropriate
covariance functions. A good resource to learn more is Rasmussen and
Williams' [book](http://www.gaussianprocess.org/gpml/). I also found the [videos](https://youtu.be/50Vgw11qn0o) by Hennig enlightening. 
