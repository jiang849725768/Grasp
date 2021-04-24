import numpy as np

qw = 0.2540757566218009
qx = -0.4015939913304674
qy = 0.7705989706898617
qz = -0.4246704632960121
x = -0.6327588252599476
y = -0.004787207347324234
z = 0.34981019783788914

rot_matrix = 2.0 * np.array(
    [[0.5 - qy**2 - qz**2, qx * qy - qz * qw, qx * qz + qy * qw, x / 2.0],
     [qx * qy + qz * qw, 0.5 - qx**2 - qz**2, qy * qz - qx * qw, y / 2.0],
     [qx * qz - qy * qw, qy * qz + qx * qw, 0.5 - qx**2 - qy**2, z / 2.0],
     [0.0, 0.0, 0.0, 0.5]])
np.set_printoptions(suppress=True)
np.savetxt('tf_1.txt', rot_matrix)
print(rot_matrix)
