import numpy as np
# np.set_printoptions(suppress=True)
# 作用是取消numpy默认的科学计数法，测试表明open3d点云读取函数没法读取科学计数法的表示
import open3d as o3d
from pandas import DataFrame

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

def line_set(points):
    np.set_printoptions(suppress=True)
    np.savetxt('temp.txt', points)
    points = np.genfromtxt("temp.txt", delimiter=" ")
    # points = points.T
    points = DataFrame(points[:, 0:3])  # 选取每一列 的 第0个元素到第5个元素   [0,3)
    points.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    # point_cloud_pynt = PyntCloud(points)  # 将points的数据 存到结构体中
    # average_point = np.array(points.mean())
    medium_point = np.array(points.median())
    # print(average_point)
    # point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # to_instance实例化

    data_mean = np.mean(points, axis=0)  # 对列求取平均值
    # 归一化
    normalize_data = points - data_mean
    # SVD分解
    # 构造协方差矩阵
    H = np.dot(normalize_data.T, normalize_data)
    # SVD分解
    eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(H)  # H = U S V
    # 逆序排列
    sort = eigenvalues.argsort()[::-1]
    # eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]

    # print(eigenvectors[:, 0])  # 最大特征值 对应的特征向量，即第一主成分
    # print(eigenvectors[:, 1])  # 第二主成分

    # 在原点云中画图
    point = [medium_point, eigenvectors[:, 0], eigenvectors[:, 1]]  # 提取第一v和第二主成分 也就是特征值最大的对应的两个特征向量 第一个点为原点
    lines = [[0, 1], [0, 2]]  # 有点构成线，[0, 1]代表点集中序号为0和1的点组成线，即原点和两个成分向量划线
    colors = [[1, 0, 0], [0, 1, 0]]  # 为不同的线添加不同的颜色
    # 构造open3d中的 LineSet对象，用于主成分的显示
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    pcd = o3d.io.read_point_cloud('temp.txt', format='xyzrgb')
    o3d.visualization.draw_geometries([pcd, line_set])

    return medium_point, eigenvectors

def main():
    item = 'apple'
    item_pc = np.load(f'{item}_pc.npy')
    item_color = np.load(f'{item}color.npy')
    item_both = np.hstack((item_pc, item_color / 255))
    line_set(item_both)
    # data = np.load('croissant_pc.npy')
    # pc_txt_data = np.savetxt('croissant_pc.txt', data)
    # data = np.load('croissantcolor.npy')
    # data = data / 255
    # color_txt_data = np.savetxt('croissantcolor.txt', data)

    # with open('croissant_pc.txt', 'r') as fa:  # 读取需要拼接的前面那个TXT
    #     with open('croissantcolor.txt', 'r') as fb:  # 读取需要拼接的后面那个TXT
    #         with open('croissant.txt', 'w') as fc:  # 写入新的TXT
    #             for line in fa:
    #                 fc.write(line.strip('\r\n'))  # 用于移除字符串头尾指定的字符
    #                 fc.write(" ")
    #                 fc.write(fb.readline())

    # pcd = o3d.io.read_point_cloud('croissant.txt', format='xyzrgb')
    # # 此处因为npy里面正好是 x y z r g b的数据排列形式，所以format='xyzrgb'
    # print(pcd)
    # #visualize()
    # o3d.visualization.draw_geometries([pcd], width=1200, height=600)  # 可视化点云

if __name__ == "__main__":
    main()
