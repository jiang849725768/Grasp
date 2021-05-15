import numpy as np

c1 = {}
c2 = {}
qw = qx = qy = qz = x = y = z = 0.


# ------------------------改动区域----------------------------------------------
cam_num = 2
yaml_num = 1

# c1['qw'] = [0.035939704309634564, 0.02383782356941807, 0.1741914899663461]
# c1['qx'] = [-0.05979474979180313, -0.0445499391679695, -0.38817245066169287]
# c1['qy'] = [0.8399441608186378, 0.840311689758471, 0.8165518413474961]
# c1['qz'] = [-0.5381697987235498, -0.5397437587808371, -0.33472779015727866]
# c1['x'] = [-0.22037375437811133, -0.1827926279854247, -0.2687896024508472]
# c1['y'] = [0.21474330127874308, 0.2395012650112968, -0.2687896024508472]
# c1['z'] = [0.43844371601506144, 0.43944570196770055, 0.4979044150317358]

c1['qw'] = [0.0515047321308]
c1['qx'] = [-0.0664174090615]
c1['qy'] = [0.850506121404]
c1['qz'] = [-0.519206440442]
c1['x'] = [-0.203384792318]
c1['y'] = [0.130079514649]
c1['z'] = [0.452954483679]

# c2['qw'] = [0.4005991079723564, 0.3925330180243339, 0.33945206307759535]
# c2['qx'] = [-0.9046560497256034, -0.8997902736379002, -0.9399420276503656]
# c2['qy'] = [0.07175140678249231, -0.18057009996944393, -0.02875281015571411]
# c2['qz'] = [0.12637057415139108, 0.060743166071301195, 0.021320352647952952]
# c2['x'] = [0.5704706794103946, 0.7532249449164439, 0.27539211073394726]
# c2['y'] = [-1.4360081330155252, -1.0296911389814647, -1.2185683078010998]
# c2['z'] = [0.410755421166672, 0.5118654216991828, 0.7413378787856314]

c2['qw'] = [0.353359494686]
c2['qx'] = [-0.933862370922]
c2['qy'] = [-0.0367746647329]
c2['qz'] = [0.0410580531036]
c2['x'] = [0.241335582408]
c2['y'] = [-1.22081509977]
c2['z'] = [0.66712417859]

# -------------------------------------------------------------

values = ['qw', 'qx', 'qy', 'qz', 'x', 'y', 'z']


for value_name in values:
    # exec(f"{value_name} = np.mean(c1['{value_name}'])")
    exec(f"{value_name} = c{cam_num}['{value_name}'][{yaml_num-1}]")

rot_matrix = 2.0 * np.array(
    [[0.5 - qy**2 - qz**2, qx * qy - qz * qw, qx * qz + qy * qw, x / 2.0],
     [qx * qy + qz * qw, 0.5 - qx**2 - qz**2, qy * qz - qx * qw, y / 2.0],
     [qx * qz - qy * qw, qy * qz + qx * qw, 0.5 - qx**2 - qy**2, z / 2.0],
     [0.0, 0.0, 0.0, 0.5]])
np.set_printoptions(suppress=True)
np.savetxt(f'tf_{cam_num}.txt', rot_matrix)
print(f"tf_{cam_num}:{rot_matrix}")
