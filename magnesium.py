import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
v = np.arange(-0.8,1,0.001)
B = 1/(1+np.exp(-0.062*(v*100-80))*1)
plt.plot(v,B)
