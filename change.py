import numpy as np

c1 = {}
c2 = {}
qw = qx = qy = qz = x = y = z = 0.


# ------------------------改动区域----------------------------------------------
cam_num = 1
yaml_num = 1

# c1['qw'] = [0.035939704309634564, 0.02383782356941807, 0.1741914899663461]
# c1['qx'] = [-0.05979474979180313, -0.0445499391679695, -0.38817245066169287]
# c1['qy'] = [0.8399441608186378, 0.840311689758471, 0.8165518413474961]
# c1['qz'] = [-0.5381697987235498, -0.5397437587808371, -0.33472779015727866]
# c1['x'] = [-0.22037375437811133, -0.1827926279854247, -0.2687896024508472]
# c1['y'] = [0.21474330127874308, 0.2395012650112968, -0.2687896024508472]
# c1['z'] = [0.43844371601506144, 0.43944570196770055, 0.4979044150317358]

c1['qw'] = [0.053797056129959685, 0.050408591875659714, 0.06029308173910366]
c1['qx'] = [-0.06525429032210257, -0.06839571970588947, -0.07728435091840172]
c1['qy'] = [0.8419115291230564, 0.8383968422658228, 0.8402021872944347]
c1['qz'] = [-0.5329472126542981, -0.5383973758012974, -0.533340564614366]
c1['x'] = [-0.17244144233643208, -0.17723122642341899, -0.1939197058398247]
c1['y'] = [0.20348711595342955, 0.21253747681484422, 0.19566640457083236]
c1['z'] = [0.4635640878339424, 0.45346050823631057, 0.4661138501569145]

# c2['qw'] = [0.4005991079723564, 0.3925330180243339, 0.33945206307759535]
# c2['qx'] = [-0.9046560497256034, -0.8997902736379002, -0.9399420276503656]
# c2['qy'] = [0.07175140678249231, -0.18057009996944393, -0.02875281015571411]
# c2['qz'] = [0.12637057415139108, 0.060743166071301195, 0.021320352647952952]
# c2['x'] = [0.5704706794103946, 0.7532249449164439, 0.27539211073394726]
# c2['y'] = [-1.4360081330155252, -1.0296911389814647, -1.2185683078010998]
# c2['z'] = [0.410755421166672, 0.5118654216991828, 0.7413378787856314]

c2['qw'] = [0.3623197907236873, 0.3668284844666865, 0.30808077638957865]
c2['qx'] = [-0.9307820697377235, -0.9294518861438904, -0.9443249079917779]
c2['qy'] = [-0.03542801100253696, -0.028620808337131445, -0.10404568678694538]
c2['qz'] = [0.0333760983505747, 0.027145969453452875, 0.050111859140800415]
c2['x'] = [0.3057633024846408, 0.2771530867088101, 0.24353178980450604]
c2['y'] = [-1.2416273519862964, -1.2416273519862964, -1.1542118883033612]
c2['z'] = [0.6802743020261058, 0.6608611812767224, 0.6471588013669721]

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
