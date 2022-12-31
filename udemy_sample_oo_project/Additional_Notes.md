
# MOMENTS
https://www.thoughtco.com/what-are-moments-in-statistics-3126234

Moments in mathematical statistics involve a basic calculation. These calculations can be used to find a probability distribution's mean, variance, and skewness.

Suppose that we have a set of data with a total of n discrete points. One important calculation, which is actually several numbers, is called the sth moment. The sth moment of the data set with values x1, x2, x3, ... , xn is given by the formula:

(x1s + x2s + x3s + ... + xns)/n

Using this formula requires us to be careful with our order of operations. We need to do the exponents first, add, then divide this sum by n the total number of data values.
A Note on the Term 'Moment'

The term moment has been taken from physics. In physics, the moment of a system of point masses is calculated with a formula identical to that above, and this formula is used in finding the center of mass of the points. In statistics, the values are no longer masses, but as we will see, moments in statistics still measure something relative to the center of the values.​
First Moment

For the first moment, we set s = 1. The formula for the first moment is thus:

(x1x2 + x3 + ... + xn)/n

This is identical to the formula for the sample mean.

The first moment of the values 1, 3, 6, 10 is (1 + 3 + 6 + 10) / 4 = 20/4 = 5.
Second Moment

For the second moment we set s = 2. The formula for the second moment is:

(x12 + x22 + x32 + ... + xn2)/n

The second moment of the values 1, 3, 6, 10 is (12 + 32 + 62 + 102) / 4 = (1 + 9 + 36 + 100)/4 = 146/4 = 36.5.
Third Moment

For the third moment we set s = 3. The formula for the third moment is:

(x13 + x23 + x33 + ... + xn3)/n

The third moment of the values 1, 3, 6, 10 is (13 + 33 + 63 + 103) / 4 = (1 + 27 + 216 + 1000)/4 = 1244/4 = 311.

Higher moments can be calculated in a similar way. Just replace s in the above formula with the number denoting the desired moment.
Moments About the Mean

A related idea is that of the sth moment about the mean. In this calculation we perform the following steps:

    First, calculate the mean of the values.
    Next, subtract this mean from each value.
    Then raise each of these differences to the sth power.
    Now add the numbers from step #3 together.
    Finally, divide this sum by the number of values we started with.

The formula for the sth moment about the mean m of the values values x1, x2, x3, ..., xn is given by:

ms = ((x1 - m)s + (x2 - m)s + (x3 - m)s + ... + (xn - m)s)/n
First Moment About the Mean

The first moment about the mean is always equal to zero, no matter what the data set is that we are working with. This can be seen in the following:

m1 = ((x1 - m) + (x2 - m) + (x3 - m) + ... + (xn - m))/n = ((x1+ x2 + x3 + ... + xn) - nm)/n = m - m = 0.
Second Moment About the Mean

The second moment about the mean is obtained from the above formula by settings = 2:

m2 = ((x1 - m)2 + (x2 - m)2 + (x3 - m)2 + ... + (xn - m)2)/n

This formula is equivalent to that for the sample variance.

For example, consider the set 1, 3, 6, 10. We have already calculated the mean of this set to be 5. Subtract this from each of the data values to obtain differences of:

    1 – 5 = -4
    3 – 5 = -2
    6 – 5 = 1
    10 – 5 = 5

We square each of these values and add them together: (-4)2 + (-2)2 + 12 + 52 = 16 + 4 + 1 + 25 = 46. Finally divide this number by the number of data points: 46/4 = 11.5
Applications of Moments

As mentioned above, the first moment is the mean and the second moment about the mean is the sample variance. Karl Pearson introduced the use of the third moment about the mean in calculating skewness and the fourth moment about the mean in the calculation of kurtosis.
