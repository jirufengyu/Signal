import numpy as np
b = np.arange(48).reshape(12,4)
a=np.arange(12).reshape(3,4)
b[2:5,]=a
print(b)