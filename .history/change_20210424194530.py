import numpy as np

qw = 0.3012059648160853
qx = -0.9399215684789926
qy = -0.14562388709953764
qz = 0.013291926009844935
x = 0.06794185288335138
y = -1.10570556302234
z = 0.6627122455341279

rot_matrix = 2.0 * np.array(
    [[0.5 - qy**2 - qz**2, qx * qy - qz * qw, qx * qz + qy * qw, x / 2.0],
     [qx * qy + qz * qw, 0.5 - qx**2 - qz**2, qy * qz - qx * qw, y / 2.0],
     [qx * qz - qy * qw, qy * qz + qx * qw, 0.5 - qx**2 - qy**2, z / 2.0],
     [0.0, 0.0, 0.0, 0.5]])
np.set_printoptions(suppress=True)
np.savetxt('tf_1.txt', rot_matrix)
print(rot_matrix)
