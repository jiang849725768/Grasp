import numpy as np
np.set_printoptions(suppress=True)
# 作用是取消numpy默认的科学计数法，测试表明open3d点云读取函数没法读取科学计数法的表示
import open3d as o3d

'''
def visualize(pointcloud):
    from open3d.open3d.geometry import PointCloud
    from open3d.open3d.utility import Vector3dVector
    from open3d.open3d.visualization import draw_geometries

    # from open3d_study import *

    # points = np.random.rand(10000, 3)
    point_cloud = PointCloud()
    point_cloud.points = Vector3dVector(pointcloud[:, 0:3].reshape(-1, 3))
    draw_geometries([point_cloud], width=800, height=600)
    return point_cloud
'''


data = np.load('croissant_pc.npy')
pc_txt_data = np.savetxt('croissant_pc.txt', data)
data = np.load('croissantcolor.npy')
data = data / 255
color_txt_data = np.savetxt('croissantcolor.txt', data)

with open('croissant_pc.txt', 'r') as fa:  # 读取需要拼接的前面那个TXT
    with open('croissantcolor.txt', 'r') as fb:  # 读取需要拼接的后面那个TXT
        with open('croissant.txt', 'w') as fc:  # 写入新的TXT
            for line in fa:
                fc.write(line.strip('\r\n'))  # 用于移除字符串头尾指定的字符
                fc.write(" ")
                fc.write(fb.readline())

pcd = o3d.io.read_point_cloud('croissant.txt', format='xyzrgb')
# 此处因为npy里面正好是 x y z r g b的数据排列形式，所以format='xyzrgb'
print(pcd)
#visualize()
o3d.visualization.draw_geometries([pcd], width=1200, height=600)  # 可视化点云
