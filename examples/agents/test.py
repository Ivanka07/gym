import numpy as np
A = np.zeros((9, 9)) 
v1 = [1.5839, -0.2615, 0.82]
v2 = [1.3039664936065674, 0.9013693177700043, 0.5243578672409058]

for i in range(3):
	for j in range(3):
		A[i * 3 + j, j * 3:(j + 1) * 3] = v1[i]

print(A)
b = np.hstack(v2)
print('b', b.shape)
M = np.linalg.solve(  A, b.reshape(3,1)).reshape((3, 3))
print('M=', M) 