import numpy as np
import cvoxels

x = np.vstack([np.random.randn(1000),np.random.randn(1000),np.random.randn(1000)]).T

U, S, V = cvoxels.test(np.linalg.svd, x)
print U
print S
print V

u, s, v = np.linalg.svd(x)

print u
print s
print v
