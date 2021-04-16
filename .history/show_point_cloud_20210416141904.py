import numpy as np
# np.set_printoptions(suppress=True)
# 作用是取消numpy默认的科学计数法，测试表明open3d点云读取函数没法读取科学计数法的表示
import open3d as o3d
from pandas import DataFrame
from sklearn.decomposition import PCA

def line_show(item_pc, item_color):
    pca = PCA(n_components=2)
    pca.fit(item_pc)
    
    # 判断特征向量效果
    # print(pca.explained_variance_ratio_) 
    

    feature_vector = pca.components_

    fv1 = np.array(feature_vector[0])
    fv2 = np.array(feature_vector[1])
    fv3 = np.cross(fv1, fv2)

    print(fv1.dot(fv2.T))

    medium_point = np.array(DataFrame(item_pc).median())
    item_both = np.hstack((item_pc, item_color / 255))
    # 取消numpy默认的科学计数法，测试表明open3d点云读取函数没法读取科学计数法的表示
    np.set_printoptions(suppress=True)
    np.savetxt('temp.txt', item_both)

    # 在原点云中画图
    point = [medium_point, medium_point + fv1,medium_point + fv2, medium_point + fv3] # 第一个点为图中的坐标原点
    lines = [[0, 1], [0, 2], [0, 3]]  # 由点构成线，[0, 1]代表点集中序号为0和1的点组成线，即原点和三个成分向量划线
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 为不同的线添加不同的颜色(RGB/255)
    # 构造open3d中的 LineSet对象，用于主成分的显示
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    pcd = o3d.io.read_point_cloud('temp.txt', format='xyzrgb')
    # o3d.visualization.draw_geometries([pcd, line_set])
    o3d.visualization.draw_geometries([pcd, line_set])

    return medium_point, [fv1, fv2, fv3]

def main():
    item = 'croissant'
    item_pc = np.load(f'{item}_pc.npy')
    item_color = np.load(f'{item}_color.npy')
    
    # line_show(item_pc, item_color)
    tf_matrix = np.loadtxt("tf.txt")
    print(tf_matrix)
    print(item_pc.shape)
    add_line = np.ones([item_pc.shape[0],1], dtype=float)
    item_pc = np.hstack((item_pc, add_line))
    print(item_pc[:5])
    # new_item_pc = tf_matrix.dot(item_pc.T)
    # line_show(new_item_pc, item_color)



if __name__ == "__main__":
    main()
