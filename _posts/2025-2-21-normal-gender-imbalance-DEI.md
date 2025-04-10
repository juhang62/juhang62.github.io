---
layout: post
title: "Gender imbalance in college enrollment and DEI's conflicted goals: insights gained from normal distributions" 
---

{% include mathjax.html %}

Tulane University leads a national trend in higher female enrollment: roughly two-thirds of the school's incoming freshman class are women ([here](https://archive.ph/H0NIZ) is an news article about it).  As a professor, I found the reality in a classroom even more stark than what the number suggests, maybe because boys skip classes more often.  For those who attend my class, I need to find a way to engage them. 

I teach an introductory statistic course. My students just learned normal distributions. It would be interesting to use normal distributions to shed light on this issue. To make the modeling accessible to the level of the students, some simplifying assumptions are needed. 

SAT scores are known to be approximately normally distributed. So, we assume that both female and male applicants’ scores follow a normal distribution but with different means and variances. Let $\mu_1, \sigma_1^2$ be the mean and variance of female applicants' score distribution and let $\mu_2, \sigma_2^2$ be the male's. To simplify the admission process, let’s just use a cut-off based on the SAT score. That is, a same cut-off is used for both female and male applicants. This is no doubt an important equity requirement. Even the university spokesman said that they did not lower admission standard for males. Since we do not know the numbers of applicants, we focus on the acceptance rate. Can the admission office pick a cut-off so that two genders have the same acceptance rate?

Students just learned standardization. It is intuitive for them to come up with an equation 

$$\begin{eqnarray}
\frac{x-\mu_1}{\sigma_1} &= & \frac{x-\mu_2}{\sigma_2}, \label{eq:eqofx}
\end{eqnarray}$$

 although not all of them explicitly mentioned the monotonicity of CDF. Here $x$ is the cut-off score. 

Then I give students some made up values of the means and variances (e.g. $\mu_1=1000,\mu_2=900,\sigma_1=200,\sigma_2=300$) and ask them to find the acceptance rate. Just plug these values into \eqref{eq:eqofx} then solve for the cut-off $x$, for which the acceptance rate turns out to be below 50%. Furthermore, it can be shown that as long as $\mu_1>\mu_2$ and $\sigma_1<\sigma_2$ (we will stick with this assumption. no offence intended to either gender), the acceptance rate must be below 50%. In this sense, inclusion and equity cannot be achieved at the same time.

The story goes on that the university wants to be more inclusive by cranking up acceptance rate. As we just learned, the gender-equal acceptance rate must be forgone. To have more than 50% acceptance rate, we must have the cut-off $x<mu_1$. It follows that 

$$\frac{x-\mu_1}{\sigma_1}  <  \frac{x-\mu_2}{\sigma_2}, $$

from which, we have reached the conclusion that the female acceptance rate will be higher than the male’s. The conclusion here may feel contrived but the experience of modeling the real world is valuable. Along the way, we even discovered that equity and inclusion are conflicted goals. What an intricate implication from such a simple model! 

I was very glad to see that many students were enthusiastic about solving this problem although most of them did not manage to finish the entire problem. To my surprise, a student solved this problem fully using a notation for standard normal CDF that I never taught in class. What is not a surprise is that the student did not give a definition of the notation. After I wrote the problem, I fed it to ChatGPT and this student’s solution looked very similar to the output of ChatGPT. I told students that I use ChatGPT too. I was once worried that my students would use ChatGPT to solve the problems I gave them. I have given up coming up with a problem that can stumble ChatGPT. However, I still use it to check if my problem makes sense and ChatGPT can help me write up the solution quickly. In the era of AI, being critical and creative is more important than ever. I tried to ask ChatGPT to write a problem of a normal distribution in the context of gender imbalances in colleges. It output a lot of general and seemly correct information but not quite usable. With some specific prompts it seemed to get better, but this requires a user to know what he or she wants.  At this point, I was relieved that I will not soon be replaced by AI but the ones who only know how to copy answers generated by AI may not be this lucky as I told my students. 
