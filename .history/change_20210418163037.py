import numpy as np

qw = 0.1720470371742105
qx = -0.9827660452159855
qy = 0.06628757102999193
qz = 0.013291926009844935
x = 0.04927139629736895
y = -0.973182372715904
z = 0.8348414060704258

rot_matrix = 2.0 * np.array(
    [[0.5 - qy**2 - qz**2, qx * qy - qz * qw, qx * qz + qy * qw, x / 2.0],
     [qx * qy + qz * qw, 0.5 - qx**2 - qz**2, qy * qz - qx * qw, y / 2.0],
     [qx * qz - qy * qw, qy * qz + qx * qw, 0.5 - qx**2 - qy**2, z / 2.0],
     [0.0, 0.0, 0.0, 0.5]])
np.set_printoptions(suppress=True)
np.savetxt('tf.txt', rot_matrix)
print(rot_matrix)
