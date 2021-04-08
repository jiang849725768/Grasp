from scipy.spatial.transform import Rotation as R
import numpy as np
print('test')
# use [:, np.newaxis] to transform from row vector to col vector
position = np.array(
    [0.03982763511251628, -0.8316034171999891, 0.8334303864998827])[:, np.newaxis].T
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

