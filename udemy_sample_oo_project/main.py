import numpy as np
from matplotlib import pyplot as plt
myarray1=np.array([2,2,3])
myarray2=np.array([2,3,4])
plt.scatter(myarray1,myarray2)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()