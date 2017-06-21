import numpy as np
from functools import reduce
from collections import defaultdict

predictions = np.zeros(5)
answers = np.array([[1.,2.,3.,4.,4.],[1.,1.,2.,3.,4.]])
b = [0.1, 1]
np.histogram(answers[:,0], weights=b)