import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import random
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import colorsys
from pandas import DataFrame
import open3d as o3d

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)
KTF.set_session(session)

ROOT_DIR = os.path.abspath("/home/jiang/Grasp")
sys.path.append(ROOT_DIR)

GRASP_DIR = os.path.join(ROOT_DIR, "UR5-control-with-RG2")
sys.path.append(GRASP_DIR)
import test_main as grasp


def MaskRCNN():
    # Root directory of the project
    # ROOT_DIR = os.path.abspath("/home/jiang/Grasp")

    # Import Mask RCNN
    # sys.path.append(ROOT_DIR)  # To find local version of the library
    # print(sys.path)
    from mrcnn.config import Config
    # from mrcnn import utils
    import mrcnn.model as modellib

    # from mrcnn import visualize

    class ShapesConfig(Config):
        NAME = "shapes"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 28  # background + 28 shapes

        IMAGE_MIN_DIM = 480
        IMAGE_MAX_DIM = 640

        RPN_ANCHOR_SCALES = (8 * 4, 16 * 4, 32 * 4, 64 * 4, 128 * 4
                             )  # anchor side in pixels

        TRAIN_ROIS_PER_IMAGE = 64

        STEPS_PER_EPOCH = 200

        VALIDATION_STEPS = 2

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = '/home/jiang/Grasp/mask_rcnn_shapes_0200.h5'

    class InferenceConfig(ShapesConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=MODEL_DIR,
                              config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    return model


def PCA(dataMat):
    meanVal = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVal
    covMat = np.cov(meanRemoved, rowvar=0)
    eigvals, eigVectors = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigvals)
    # print('特征值：', eigvals)
    # print('特征值顺序：', eigValInd)
    eigValInd = eigValInd[1]
    # print(eigValInd)
    redEigVects = eigVectors[:, eigValInd]
    # print('特征向量：', redEigVects)
    # print('type : ', redEigVects.dtype)
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVal
    return lowDDataMat, reconMat


__k = 0


def detect_objects_in_image(image, model):
    # # choose your color

    hsv = [(50 / 81, 1, 1) for i in range(81)]
    COLOR = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(COLOR)

    # Root directory of the project
    # ROOT_DIR = os.path.abspath("/home/jiang/Grasp")

    # Import Mask RCNN
    # sys.path.append(ROOT_DIR)  # To find local version of the library

    from mrcnn import visualize

    class_names = [
        'BG', "red_pepper", 'green_pepper', "carrot", "turnip", "eggplant",
        "baozi", "croissant", "cupcake", "ginger", "cake", "corn", "grape",
        "banana", "kiwi", "lemon", "pear", "apple", "carambola", "train",
        "detergent", "plate_w", "plate_g", "paper_box", "plastic_box", "cup_",
        "mouse", "hand", 'watermelon'
    ]

    # graspable_class = ['red_pepper', 'green_pepper', 'carrot', 'turnip', 'eggplant',
    #                    'baozi', 'croissant', 'cupcake', 'ginger', 'cake', 'corn', 'grape',
    #                    'banana', 'kiwi', 'lemon', 'pear', 'apple', 'carambola',
    #                    'train', 'detergent', "mouse", 'watermelon', 'cup_']

    image = image[..., ::-1]
    # Run detection
    s_time = time.time()
    results = model.detect([image], verbose=1)
    e_time = time.time()
    print('mask rcnn using : ',
          int(round(e_time * 1000)) - int(round(s_time * 1000)))
    # Visualize results
    r = results[0]
    global __k
    save_image = image[..., ::-1]
    cv2.imwrite('/home/jiang/Grasp/result/res' + str(__k) + '.jpg', save_image)
    __k += 1
    RESULT_SAVE_PATH = "/home/jiang/Grasp/result"
    # !!! change the function to choose which object to show in the results
    save_path_name = os.path.join(RESULT_SAVE_PATH, 'img.jpg')

    # 显示掩码图片
    # masked_image = visualize.display_instances(save_path_name, image, r['rois'], r['masks'], r['class_ids'],
    #                                            class_names, r['scores'], colors=COLOR)

    print(r['masks'].shape)
    target_object_dict = {}
    for score, index, cood, index_i in zip(r['scores'],
                                           r['class_ids'], r['rois'],
                                           range(len(r['class_ids']))):
        mask_points = []
        mask = r['masks'][:, :, index_i]
        print(class_names[index])

        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x][y] != False:
                    mask_points.append([x, y])
        mask_points = np.array(mask_points)
        if mask_points.shape[0] < 1500:
            continue
        lowd, rec = PCA(mask_points)

        center_point = mask_points.mean(axis=0, keepdims=True)

        tan_theta = (rec[rec.shape[0] - 1, 0] -
                     rec[0, 0]) / (rec[rec.shape[0] - 1, 1] - rec[0, 1])
        theta = -np.arctan(tan_theta)
        # print('pca theta : ', theta / np.pi * 180)
        # print('mask center point : ', center_point)
        # np.save(class_names[index]+'_mask.npy', mask_points)
        target_x_in_pixel = center_point[0][0]
        target_y_in_pixel = center_point[0][1]
        # print('target in image (use center point) :', target_x_in_pixel, target_y_in_pixel)
        p_0 = np.array([-33.54, -767.63])
        p_1 = np.array([-26.01, -447.32])
        p_2 = np.array([293.13, -449.37])

        x_dist = (p_1[1] - p_0[1]) / 300
        y_dist = (p_2[0] - p_1[0]) / 300

        obj_x = (target_x_in_pixel - 150) * x_dist + p_0[0]
        obj_y = (target_y_in_pixel - 150) * y_dist + p_0[1]

        # print(class_names[index], '(x,y):', obj_x, obj_y)
        #target_object_dict[class_names[index]] = [obj_x / 1000, obj_y / 1000, theta, mask_points]
        target_object_dict[class_names[index]] = [mask_points]
    return target_object_dict


def save_objects_point_cloud(total_point_cloud, color_img,
                             target_objects_dict):
    np.save('full_point_cloud_test', total_point_cloud)
    np.save('full_color', color_img)
    object_dict = {}
    for name, value in target_objects_dict.items():
        mask = value[-1]
        object_pc = np.zeros((mask.shape[0], 3), dtype=float)
        object_color = np.zeros((mask.shape[0], 3), dtype=np.uint8)
        for point_index in range(mask.shape[0]):
            object_pc[point_index] = list(total_point_cloud[
                mask[point_index][0]][mask[point_index][1]][0])
            # print(object_pc[point_index])
            # input('>>>>>>>>>>>>>>>>>>>>>>>>>')
            object_color[point_index] = list(
                color_img[mask[point_index][0]][mask[point_index][1]])
        object_both = np.hstack((object_pc, object_color / 255))
        medium_points, eigenvectors = line_set(object_both)
        object_dict[name] = [
            medium_points, eigenvectors[:, 0], eigenvectors[:, 1], object_both
        ]
        np.save(name + '_pc', object_pc)
        np.save(name + 'color', object_color)
    # print(object_dict)

    return object_dict


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
    # point = [medium_point, eigenvectors[:, 0], eigenvectors[:, 1]]  # 提取第一v和第二主成分 也就是特征值最大的对应的两个特征向量 第一个点为原点
    # lines = [[0, 1], [0, 2]]  # 有点构成线，[0, 1]代表点集中序号为0和1的点组成线，即原点和两个成分向量划线
    # colors = [[1, 0, 0], [0, 1, 0]]  # 为不同的线添加不同的颜色
    # # 构造open3d中的 LineSet对象，用于主成分的显示
    # line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point), lines=o3d.utility.Vector2iVector(lines))
    # line_set.colors = o3d.utility.Vector3dVector(colors)
    # pcd = o3d.io.read_point_cloud('temp.txt', format='xyzrgb')
    # o3d.visualization.draw_geometries([pcd, line_set])

    return medium_point, eigenvectors


def go_to_dot(dot):

    move_dot = np.append(dot, 1.0)

    tf = np.loadtxt('tf.txt')
    new_dot = tf.dot(move_dot)
    current_tcp = grasp.get_current_tcp()
    print(f"current_tcp:{current_tcp}")
    move_tcp = np.hstack((new_dot[:3], current_tcp[3:]))
    print(f"move_tcp:{move_tcp}")
    grasp.move_to_tcp(move_tcp)


def main():
    # input('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # Load model
    model = MaskRCNN()

    # Set the Camera
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    # Start streaming
    pipeline.start(config)

    align = rs.align(rs.stream.color)
    pc = rs.pointcloud()

    for i in range(20):
        # print('Camera heating! Wait for a second!')
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
    print('Camera heating over.')

    for i in range(3):
        print(f'----------第{i+1}张照片-----------')
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth = frames.get_depth_frame()
        color = frames.get_color_frame()

        img_color = np.asanyarray(color.get_data())
        img_depth = np.asanyarray(depth.get_data())
        pc.map_to(color)
        points = pc.calculate(depth)
        vtx = np.asanyarray(points.get_vertices())
        vtx = np.reshape(vtx, (720, 1280, -1))

        # get 3d point cloud (masked)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        rgb_image = np.asanyarray(color_frame.get_data())
        rgb_image[:, :, [0, 2]] = rgb_image[:, :, [2, 0]]
        # print(rgb_image.shape)

        # 显示rgb图片
        # plt.imshow(rgb_image)

        # plt.pause(1)  # pause 1 second
        # plt.clf()
        target_objects_dict = detect_objects_in_image(img_color, model)

        object_dict = save_objects_point_cloud(vtx, rgb_image,
                                               target_objects_dict)
        '''
        if 'eggplant' in object_dict:
            item_dot = object_dict['eggplant'][0]
            print(item_dot)
            go_to_dot(item_dot)
            '''
        test_object = ['eggplant', 'carrot', 'grape']
        for item in test_object:
            if item in object_dict:
                item_dot = object_dict[item][0]
                print(item_dot)
                go_to_dot(item_dot)

    grasp.go_home()

if __name__ == "__main__":
    main()
