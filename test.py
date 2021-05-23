# import tensorflow as tf
# # 检查tensorflow是否得到CUDA支持，安装成功则显示true，否则为false
# # print(tf.test.is_built_with_cuda())
# # 检查tensorflow是否可以获取到GPU，安装成功则显示true，否则为false
# print(tf.test.is_gpu_available())
import open3d as o3d
import numpy as np


def line_set_show(medium_point, fv1, fv2, fv3):
    '''展示点云及位姿坐标系'''

    # 在原点云中画图
    # 第一个点为图中的坐标原点
    point = [
        medium_point, medium_point + fv1, medium_point + fv2,
        medium_point + fv3, [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ]
    # 由点构成线，[0, 1]代表点集中序号为0和1的点组成线，即原点和三个成分向量划线
    lines = [[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [4, 7]]
    # 为不同的线添加不同的颜色(RGB/255)
    # 红1绿2蓝3
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # 构造open3d中的 LineSet对象，用于主成分的显示
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([line_set])


if __name__ == "__main__":
    matrix = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
              [0., 0., 0., 1.]]
    np.set_printoptions(suppress=True)
    tf_matrix = np.loadtxt("tf_2.txt")
    print(tf_matrix)
    new_matrix = tf_matrix.dot(matrix)
    medium_point = new_matrix.copy().T[3][:3]
    rotation_matrix = np.array(new_matrix[:3, :3])
    feature_vector = rotation_matrix.copy().T
    fv1, fv2, fv3 = feature_vector[0], feature_vector[1], feature_vector[2]
    line_set_show(medium_point, fv1, fv2, fv3)
