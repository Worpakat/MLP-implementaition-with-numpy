import numpy as np

a = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])

b = np.array([[2.0, 4.0, 1.0],
              [1.0, 3.0, 2.0]])
a += b * 0.1

print(a)

a += b * 0.1
print(a)