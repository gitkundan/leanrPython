~ Read from page 21


# What is Probability
Equally likely (like randint) then distribution will be uniform
natural world where tendency towards central : normal distribution

two ways of thinking:
a) actual chances : frequentist (p-value, CI) - objective 
b) belief on how strongly it will occur : Bayesian - subjective (Bayes factors);
    assign probability based on beliefs, good for infrequent events (earthquake)

What causes uncertainty:
a) Noise in data : incorrect measurement / ommission / mistyping
b) incomplete coverage of domain / not random sampling
c) imperfect model : all models are wrong but some are useful ; map is not territory
d) Useful for capturing conditional dependency among variables

# Terminology
one observation = one example = one instance = one row
discrete variable : countable noun
continuous variable : between a range, not discrete
 

# Types
a) Joint(hungry ∩ Angry) : p of two or more events occuring simultaneously e.g. city 1=Sunny ∩ city 2=Sunny
b) Marginal/Standalone : p of event irrespective of the outcome of other e.g. city 1 = Sunny irrespective of city 2 weather i.e. row total and column total
c) Conditional : p of event B given event A has happened i.e. city 1 = Sunny given city 2 = Sunny

pdf : For a random variable x, P(x) is a function that assigns a probability to all possible values of x.

The probability of a specific event A for a random variable x is denoted as P(x = A), or simply as P(A).


Probability of 2 random variables:
As such, there are three main types of probability we might want to consider; they are:
a) Joint Probability (INNER JOIN::**Prodcut Rule**): Probability of events A AND B. :: p(hungry ∩ Angry) = p(conditional hungry given angry) * marginal p(A) :::: PRODUCT RULE

b) Marginal Probability (**Sum Rule**): Probability of event A given variable Y i.e. fixed value of Y. Probability animal is frog DOES NOT matter whether the forest is temperate or tropical or tundra = Probability animal is frog given forest is temperate + Probability animal is frog given forest is tropical + Probability animal is frog given forest is tundra
the sum or union over all the probabilities of all events for the second variable for a given fixed event for the first variable. 
P(X is A) = sum over[P(X = A, Y = y)] where y∈Y :::: SUM RULE


c) Conditional Probability (TIT for TAT **P(A|B) corollary from product rule**): Probability of event A GIVEN event B.
The conditional probability for events A given event B is calculated as follows:
P(A|B) = P(A ∩ B) / P(B)
This calculation assumes that the probability of event B is not zero, e.g. is not impossible. The notion of event A given event B does not mean that event B has occurred (e.g. is certain); instead, it is the probability of event A occurring after or in the presence of event B for a given trial. This is a corollary of the PRODUCT RULE

These types of probability form the basis of much of predictive modeling with problems such as classification and regression. For example:
a) The probability of a row of data is the joint probability across each input variable. P(A and B) = P(A ∩ B) = P(A, B). The joint probability for events A and B is calculated as the probability of event A given event B multiplied by the probability of event B. This can be stated formally as follows:
P(A ∩ B) = P(A given B) × P(B)
(4.10)
The calculation of the joint probability is sometimes called the fundamental rule of probability or the product rule of probability. Here, P(A given B) is the probability of event A given that event B has occurred, called the conditional probability, described below. The joint probability is symmetrical, meaning that P(A ∩ B) is the same as P(B ∩ A).


b) The probability of a specific value of one input variable is the marginal probability across the values of the other input variables.
c) The predictive model itself is an estimate of the conditional probability of an output given an input example.

city 1 and city 2 joint probabilty distribution table is the table given as example in book
from here marginal probability = row total and column total
conditional probability will need to be calculated by calculation

P(A|B) ̸= P(B|A)

Repeat joint,marginal,distribution

For probability questions focus particularly on word choices; see what is not possible - is the sample size reducing due to choice of words

For probability questions dont follow intuitions, list the event space and apply the conditional probability formulas

**TO REVISE** Page 38 - Boy or Girl Problem

================================================================================
Distributions
Relationship between the outcome of a r.v. and its probability is distribution

Type of distribution depends on type of r.v. - continuous or discrete
this determines how the probability can be summed or how most likely estimator can be calculated

a) relationship between POSSIBLE values and their p ==> p distribution (??)
b) relationship between values to their p ==> pdf or mass function
c) relationship between outcomes <= value ==> cum distribution

Notation :
r.v. ==> X
outcome ==> x1, x2, x3

discrete X : finite set of states
boolean x : true or false
continuous : range e.g. height

Moments 
first moment of numbers (xbar) is mean
FIRST moment of differences (x-xbar) is zero
SECOND moment of differences is variance
THIRD moment of differences is skewness
FOURTH moment of differences is kurtosis

================================================================================
Discrete distribution
e.g. dice roll is discrete uniform distribution
single coin flip is Bernoulli distribution
car color is Multinoulli distribution
Poisson distribution (hospital ER arrival per unit of time)
binary classification problems
multiclass classification problems
distribution of words in text

E.V (central tendency) = mode
PMF = p for a specific discrete value
Sum(PMF for all values) = 1
CDF = p <= specific discrete value

**ONE INSTANCE VS SEQUENCE**
*Binary Random Variable: x ∈ {0, 1}; Single : Bernoulli; sequence : binary*
*Categorical Random Variable: x ∈ {1, 2, · · · , K}; Single : Multinoulli; sequence : Multinomial*
*START HERE*
one instance of coin flip is bernoulli
sequence of coin flips : calculate number of heads (i.e. success) is binomial
feed the total number of trials and p(success) of individual Bernoulli as parameters to numpy.random import binomial
Examples:
# result of flipping a coin 10 times and getting success, tested 1000 times.
>>> n, p = 10, .5  # number of trials in each experiment is n, probability of success of each trial is p
>>> s = np.random.binomial(n, p, 1000)
return value is darray (if number of experiment i.e. trial is specified) or scalar (if number of experiment is not specified i.e trial =1)
trial = number of times the experiment is repeated i.e. number of simulations
each experiment will be tossing coin 10 times (REMEMBER : bionmial implies sequence of toss where each toss is bernoulli)
1 trial is 1 life - do everything you can and simulate reality completely
more lives/simulations means more trials

Drawn samples from the parameterized binomial distribution, where each sample is equal to the number of successes over the n trials.

A company drills 9 wild-cat oil exploration wells, each with an estimated probability of success of 0.1. All nine wells fail. What is the probability of that happening?
Let's do 20,000 trials of the model, and count the number that generate zero positive results.
ez
>>> sum(np.random.binomial(9, 0.1, 20000) == 0)/20000.
# answer = 0.38885, or 38%.

Multinoulli/Cateegorical Distribution
event will have one of k possible outcomes
    x ∈ {1, 2, 3, · · · , K}    

generalization of bernoulli where the possible outcomes are more than binary
e.g. dice roll hase 6 possible outcomes
e.g. classify species of iris flower into sentosa, verosa,etc. [multiclass classification]

sequence of multinoulli -> multinomial distribution
e.g. sequence of dice rolls
word frequency in NLP

================================================================================
Continuous distribution
PDF (NOT PMF) : p for a value i.e. probability frequency distribution (PFD)
**UNLIKE DISCRETE R.V. p for a given continuous r.v. CANNOT be directly specified**
**WE CAN ONLY CALCULATE AREA UNDER CURVE (integral) for a small distance either way of the actual value , so we cant say probability of height=10 but probability of height <10 or height between 10 and 20 given height follows normal distribution with mean of 5 and s.d of 2**
**More : mathinsight.org/probability_density_function_idea**

CDF = p of a value <= given outcome

PPF: Percent-Point Function, returns a discrete value that is less than or equal to
the given probability.

CDF returns PROBABILITY; PPF returns VALUE

Gaussian distribution : height of people, scores on test, weight of babies
Exponential distribution : few outcomes are most likely with a rapid decrease in
probability for all other outcomes
e.g. time until default of loan
Pareto distribution : income level
movies being a hit = pareto distribution

_______________________________________________________
Gaussian distribution
normal distrubution can be specified only **WITH 2 parameters**:
mean, variance
A normal distribution with a mean of zero and a standard deviation of 1 is called a standard normal distribution, and often data is reduced or standardized (**data coercion**) to this for analysis for ease of interpretation and comparison.

**VVI TO REMEMBER:**
a) if you just generate sample values from distribution and graph then there is no pattern : it will be random values around the mean. 
e.g. heights of children in sharks class is hovering around mean

**VVI** Understand whether the ask is to calculate p or actual values from dist

code:

#generate 10 samples from normal distribution
from numpy.random import normal
mu=50
sigma=5
n=10
sample=normal(mu,sigma,n)
print(sample)

b) if you calculate p for certain values then PDF will be familiar bell curve:
e.g. 
from scipy.stats import norm
from matplotlib import pyplot as plt
mu=50
sigma=5
dist=norm(mu,sigma)
values=[value for value in range(30,70)]
p=[dist.pdf(value) for value in values]
plt.plot(values,p)
plt.show()

c) CDF of normal curve is sigmoid (s-shaped,spoon-shaped)
In the above code change to :
p=[dist.cdf(value) for value in values]

d) calculate the ppf to get the value for the define probability
from scipy.stats import norm
from matplotlib import pyplot as plt

mu=50
sigma=5
dist=norm(mu,sigma)
low_end=dist.ppf(0.025)
high_end=dist.ppf(0.975)
print(f'Middle 95% between {low_end:2f} and {high_end:2f}')

e) frequency distribution (histogram) will be bell shaped
but not to confuse with pdf which is also bell shaped


Exponential distribution : few outcomes are most likely with a rapid decrease in
probability for all other  i.e. multi-modal
e.g. time until default of loan
time between clicks on a geiger counter
time until failure of a part

Pareto Distribution / power law distribution
80% of the events are covered by 20% of the outcomes
e.g. income of households in a country
sale of books
AFC Bournemouth team goals by players

defined by only one parameter - shape(alpha) : the steepness of the 
decrease in p; alpha is usually between 1 - 3; 1.161 is the 80-20 distn

-----------------------------------------------
10. Estimating probability distribution from a sample

The overall shape of the probability density is distribution
calculation of p for a specific outcome of r.v. is pdf

Steps in estimating
a) draw histogram : experiment different bin sizes to get multiple perspectives
                    on the same data. more the bin more fine i.e. less coarse

Reviewing a histogram with different bin sizes will help to identify whether density
looks like a common distribution or not. In most cases, you will see a unimodal
distribution, such as the familiar bell shape of the normal, the flat shape of the 
uniform, or the descending / ascending shape of exponential or Pareto


b) from the histogram shape identify the distribution and estimate the parameters (density estimation)
e.g. for normal only mean and variance are parameters, these parameters can be calculated from the sample

    from matplotlib import pyplot as plt
    from statistics import NormalDist,mean,stdev
    from scipy.stats import norm

    # #draw a sample from the population
    file=r"C:\Users\Dell\Pictures\ControlCenter4\Scan\SOCR-HeightWeight.csv"
    import pandas as pd
    df=pd.read_csv(file)
    sample=df['Height(Inches)']

    #define the sample distribution
    dist_sample=NormalDist(mean(sample),stdev(sample))

    #sample probabilities for range of outcomes
    values=[value for value in range(60,75)]
    p=[dist_sample.pdf(value) for value in values]

    #plot the histogram of the drawn sample and 
    #Importantly, we can convert the counts or frequencies 
    # in each bin of the histogram to a normalized probability 
    # to ensure the y-axis of the histogram matches the y-axis 
    # of the line plot. This can be achieved by setting 
    # the density argument to True in the call to hist().
    plt.hist(sample,bins=10,density=True) 
    plt.plot(values,p)
    plt.show()

...

page 81 : 10.5 Nonparametric Density Estimation
bimodal / multimodal distribution : so cant estimate with usual parametric
probability distribution : use nonparametric distribution i.e. all observation
considered as parameters
kernel density estimation / kernel smoothing : 
can be thought as smoothed histogram
calculate distribution at each datapoint (using a function/kernel/Gaussian) and average them to get a smooth curve

a) kernel / basis function : is a probability function; can generate probability of x (new point)
        will weight the contribution of observations based on distance to x

b) smoothing parameter / bandwith : parameter that controls the number of samples or window of samples used to estimate the probability for a new point
experiment with different window sizes and different contribution functions
and evaluate results against histograms of the data

-----------------------------------------------

page 86 : MLE (frequentist / non-Bayesian) : like KDE it is a density estimation solution; similar Bayesian one is called Max a Posteriori (MAP)
given a known distribution with known parameters, prob function = conditional p of 
    observing sample

MLE cost function gives joint p of set of given observations if underlying each
observation is i.i.d 
MLE assumption : alll solutions are equally likely

MLE treats the problem as an optimization problem => seek a set of parameters that results in the best fit for the join prob of the  data sample (X)
MLE function = P(X|θ) where X=[x1,x2,x3.....]
since θ is not r.v. more accurately: Likelyhood function L(X; θ) or P(x1, x2, x3, · · · , xn; θ)

goal seek on θ to minimise the log function (minimise negative log likelihood)

Most ML models can be framed as MLE

Chapter 12 : Page 94 [Linear Regression with MLE]
Parameters can be estimated either OLS (analytical method) or MLE (probabilistic) : both solutions will be same
MLE approach (negative log likelihood) is used to automatically find the distribution and parameters that 
best describe the data

Some notations:
prediction => yhat

yhat=model(X) is the algebraic representation of the linear regression
model parameters i.e.   (a) coefficients (beta) per input
                        (b) intercept per bias

          yhat = β0 + β1 × x1 + β2 × x2 + · · · + βm × xm    

(VVI) : Linear Regeression assumes a gaussian distribution of prediction (NOT INPUTS)
the model considers noise only in the target value of the training example and does not consider noise in the attributes describing the instances themselves

MLE approach : search procedure to maximize the log likelihood function,
or minimize -ve log likelihood (preferred)
============Logistic Regression with MLE====================
binary classification of a class label (not numerical value like linear regression): is the cat red or green; true/false

The nature of the dependent variables differentiates regression and classification problems. Regression problems have continuous and usually unbounded outputs. An example is when you’re estimating the salary as a function of experience and education level. On the other hand, classification problems have discrete and finite outputs called classes or categories. For example, predicting if an employee is going to be promoted or not (true or false) is a classification problem.

There are two main types of classification problems:

    Binary or binomial classification: exactly two classes to choose between (usually 0 and 1, true and false, or positive and negative)
    Multiclass or multinomial classification: three or more classes of the outputs to choose from


logistic function a.k.a sigmoid curve a.k.a link function

This link function looks like a really wide S. It takes in any values given from multiplying the model coefficients times a row of customer data, and it outputs a number between 0 and 1. But why does this odd function look like this?
Well, just round e to 2.7 real quick and think about the case where the input to this function is pretty big, say 10. Then the link function is:
exp(x)/(1 + exp(x)) = 2.7^10 / (1+ 2.7^10) = 20589/20590
Well, that’s basically 1, so we can see that as x gets larger, that 1 in the denominator just doesn’t matter much. But as x goes negative? Look at -10:
exp(x)/(1 + exp(x)) = 2.7^-10 / (1+ 2.7^-10) = 0.00005/1.00005

Well, that’s just 0 for the most part. In this case the 1 in the denominator means everything and the teeny numbers are more or less 0s

The sigmoid function has values very close to either 0 or 1 across most of its domain. This fact makes it suitable for application in classification methods.
``
$$
f(x) = d\xi

$$
``
Next steps:
https://realpython.com/logistic-regression-python/
data smart





θ is MLE parameter that defines botht the choice of the pdf and the parameters
of that distribution


Paused at page 111 and revision started




-----------------------------------------------

Start from
Page 79 : do the same normal distribution coding with the height dataset

Revisit this after 1 week
Start from Page 53 - "We can calculate the moments of this distribution"

Revisit after 1 week
9.4 Exponential Distribution
