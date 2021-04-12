import numpy as np

qw = 0.29451062269665734
qx = -0.9556078873666751
qy = 0.004068167532089444
qz = -0.007778736038575677
x = 0.018573537961287336
y = -0.9998781315296043
z = 0.5622653639161477

rot_matrix = 2.0 * np.array(
    [[0.5 - qy**2 - qz**2, qx * qy - qz * qw, qx * qz + qy * qw, x / 2.0],
     [qx * qy + qz * qw, 0.5 - qx**2 - qz**2, qy * qz - qx * qw, y / 2.0],
     [qx * qz - qy * qw, qy * qz + qx * qw, 0.5 - qx**2 - qy**2, z / 2.0],
     [0.0, 0.0, 0.0, 0.5]])
np.savetxt('tf.txt', rot_matrix)
print(rot_matrix)
