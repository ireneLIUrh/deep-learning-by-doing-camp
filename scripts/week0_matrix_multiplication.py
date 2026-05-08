import numpy as np

A = np.array([
    [1, 2],
    [3, 4]
])

x = np.array([
    [5],
    [6]
])

y = A @ x

print("A =")
print(A)

print("x =")
print(x)

print("A @ x =")
print(y)