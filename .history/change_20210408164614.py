from math import cos, sin, pi, atan2, asin

#四元数q=(q0,q1,q2,q3)到旋转矩阵
q0, q1, q2, q3 = 0.1225787223613701, -0.7441192315128069, 0.642821221041933, 0.1343201544638865
r11, r12, r13 = 1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (
    q1 * q3 + q0 * q2)
r21, r22, r23 = 2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (
    q2 * q3 - q0 * q1)
r31, r32, r33 = 2 * (q1 * q3 - q0 * q2), 2 * (
    q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)
print('四元数q=({0:f},{1:f},{2:f},{3:f})转换为旋转矩阵'.format(q0, q1, q2, q3))
print('{0:f} , {1:f} , {2:f}'.format(r11, r12, r13))
print('{0:f} , {1:f} , {2:f}'.format(r21, r22, r23))
print('{0:f} , {1:f} , {2:f}'.format(r31, r32, r33))
print()

from scipy.spatial.transform import Rotation as R
import numpy as np
print('test')
# use [:, np.newaxis] to transform from row vector to col vector
position = np.array(
    [0.03982763511251628, -0.8316034171999891, 0.8334303864998827])[:,
                                                                    np.newaxis]
share_vector = np.array([0, 0, 0, 1], dtype=float)[np.newaxis, :]
print('share_vector:\n', share_vector)
print('position:\n', position)
r = R.from_quat([
    -0.716556549511624, -0.6971278819736084, -0.010016582945017661,
    0.02142651612120239
])
r.as_matrix()
print('rotation:\n', r.as_matrix())
rotation_matrix = r.as_matrix()
print(rotation_matrix)

#combine three matrix or vector together
m34 = np.concatenate((rotation_matrix, position), axis=1)
print(m34)
m44 = np.concatenate((m34, share_vector), axis=0)
# m44 = np.hstack((m34, share_vector))

print(m44)

rot_vec = r.as_rotvec()
print('rot_vec:\n', rot_vec)
rot_euler = r.as_euler('zyx', degrees=False)
print('rot_euler:\n', rot_euler)

r = R.from_matrix(rotation_matrix)
print('as_quat():\n', r.as_quat())
print('as_rotvec():\n', r.as_rotvec())
print('as_euler():\n', r.as_euler('zyx', degrees=True))
