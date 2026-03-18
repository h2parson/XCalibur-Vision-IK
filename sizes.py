import numpy as np
import sys

data = np.load("./knife_data.npz")
print(data.files) 
tip_q1 = data['tip_q1']
tip_q2 = data['tip_q2']
yaw_indices = data['yaw_indices']
ratios1 = data['ratios1']
ratios2 = data['ratios2']

print("shape of tip_q1 = ", np.shape(tip_q1))
print("type of elements of tip_q1 = ", type(tip_q1[0]))
print("shape of tip_q2 = ", np.shape(tip_q2))
print("shape of yaw_indices = ", np.shape(yaw_indices))
print("type of elements of yaw_indices = ", type(yaw_indices[0]))
print("shape of ratios1 = ", np.shape(ratios1))
print("type of elements of ratios1 = ", type(ratios1[0][0]))
print("shape of ratios2 = ", np.shape(ratios2))


'''
['tip_q1', 'tip_q2', 'yaw_indices', 'ratios1', 'ratios2']
shape of tip_q1 =  (5,)
type of elements of tip_q1 =  <class 'numpy.ndarray'>
shape of tip_q2 =  (5,)
shape of yaw_indices =  (165,)
type of elements of yaw_indices =  <class 'numpy.float64'>
shape of ratios1 =  (164, 4)
type of elements of ratios1 =  <class 'numpy.float64'>
shape of ratios2 =  (164, 4)

Everything is float64

5 element array tip_q1
5 element array tip_q2
N element array yaw_indices
(N-1) x 4 element array ratios1
(N-1) x 4 element array ratios2

N is the number of sample points
'''
