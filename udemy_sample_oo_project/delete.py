file=r"C:\Users\Dell\Pictures\ControlCenter4\Scan\SOCR-HeightWeight.csv"
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from scipy.stats import norm
'''
              Index  Height(Inches)  Weight(Pounds)
count  25000.000000    25000.000000    25000.000000
mean   12500.500000       67.993114      127.079421
std     7217.022701        1.901679       11.660898
min        1.000000       60.278360       78.014760
25%     6250.750000       66.704397      119.308675
50%    12500.500000       67.995700      127.157750
75%    18750.250000       69.272958      134.892850
max    25000.000000       75.152800      170.924000
'''

df=pd.read_csv(file)
data=df['Height(Inches)']
# mu=statistics.mean(data)
# sigma=statistics.variance(data)
# dist=norm(mu,sigma)
# values=[value for value in range(60,75)]
# p=[dist.pdf(value) for value in values]
# plt.plot(values,p)
plt.hist(data,bins=100)

plt.show()