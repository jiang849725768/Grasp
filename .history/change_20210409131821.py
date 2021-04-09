from scipy.spatial.transform import Rotation as R
import numpy as np
print('test')
# use [:, np.newaxis] to add a new axis
position = np.array(
    [0.03982763511251628, -0.8316034171999891, 0.8334303864998827])[:, np.newaxis]
share_vector = np.array([0, 0, 0, 1], dtype=float)[np.newaxis, :]
print('position:\n', position)
r = R.from_quat([0.1225787223613701, -0.7441192315128069, 0.642821221041933, 0.1343201544638865])
rotation_matrix = r.as_matrix()
print(rotation_matrix)

#combine three matrix or vector together
m34 = np.concatenate((rotation_matrix, position), axis=1)
m44 = np.concatenate((m34, share_vector), axis=0)
# m44 = np.hstack((m34, share_vector))

print(m44)

quat = np.array([0.1225787223613701, -0.7441192315128069, 0.642821221041933, 0.1343201544638865])
def quaternion_to_rotation_matrix(quat):
    q = quat.copy()
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(4)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], 0.0],
         [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], 0.0],
         [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=q.dtype)
    return rot_matrix

print(quaternion_to_rotation_matrix(quat))

qw=0.1225787223613701
qx=-0.7441192315128069
qy=0.642821221041933
qz=0.1343201544638865
x=0.03982763511251628
y=-0.8316034171999891
z=0.8334303864998827
rot_matrix = 2.0*np.array(
        [[0.5 - qy**2 - qz**2, qx*qy-qz*qw, qx*qz+qy*qw, x,
         [qx*qy+qz*qw, 0.5 - qx**2-qz**2, qy*qz-qx*qw, y],
         [qx*qz-qy*qw, qy*qz+qx*qw, 0.5-qx**2-qy**2, z],
         [0.0, 0.0, 0.0, 1.0]])
print   (rot_matrix)