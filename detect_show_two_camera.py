import colorsys
import time
import random
import numpy as np
import pyrealsense2 as rs
from pandas import DataFrame

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from sklearn.decomposition import PCA

import os
import sys
# 防止ros中python2的opencv干扰导入
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

# import random
# import matplotlib.pyplot as plt
import open3d as o3d

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)
KTF.set_session(session)

ROOT_DIR = os.path.abspath("/home/jiang/Grasp")


def MaskRCNN():
    # Root directory of the project
    # ROOT_DIR = os.path.abspath("/home/jiang/Grasp")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    # print(sys.path)
    # from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn.config import Config

    # from mrcnn import visualize

    # sys.path.append(os.path.join(ROOT_DIR, "src/coco/"))
    # # To find local version
    # import coco

    class ShapesConfig(Config):
        NAME = "shapes"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 22  # background + 28 shapes

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
    COCO_MODEL_PATH = '/home/jiang/Grasp/mask_rcnn_shapes_0100.h5'

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


__k = 0


def detect_items_in_image(image, model):
    # # choose your color

    hsv = [(50 / 81, 1, 1) for i in range(81)]
    COLOR = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(COLOR)

    # Root directory of the project
    # ROOT_DIR = os.path.abspath("/home/jiang/Grasp")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library

    from mrcnn import visualize

    class_names = [
        'BG', "red_pepper", 'green_pepper', "carrot", "turnip", "eggplant",
        "baozi", "croissant", "cupcake", "ginger", "cake", "corn", "grape",
        "banana", "kiwi", "lemon", "pear", "apple", "carambola", "train",
        "detergent", "plate_w", "plate_g", "paper_box", "plastic_box", "cup_",
        "mouse", "hand", 'watermelon'
    ]

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
    visualize.display_instances(save_path_name,
                                image,
                                r['rois'],
                                r['masks'],
                                r['class_ids'],
                                class_names,
                                r['scores'],
                                colors=COLOR)

    print(r['masks'].shape)
    target_item_dict = {}
    for score, index, cood, index_i in zip(r['scores'],
                                           r['class_ids'], r['rois'],
                                           range(len(r['class_ids']))):
        mask_points = []
        mask = r['masks'][:, :, index_i]
        print(class_names[index])

        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x][y]:
                    mask_points.append([x, y])
        mask_points = np.array(mask_points)
        if mask_points.shape[0] < 1500:
            continue
        target_item_dict[class_names[index]] = [mask_points]

    return target_item_dict


def save_items_point_cloud(total_point_cloud, color_img, target_items_dict,
                           camera_num):
    # np.save('full_point_cloud_test', total_point_cloud)
    # np.save('full_color', color_img)
    item_dict = {}
    for name, value in target_items_dict.items():
        mask = value[-1]
        item_pc = np.zeros((mask.shape[0], 3), dtype=float)
        item_color = np.zeros((mask.shape[0], 3), dtype=np.uint8)
        for point_index in range(mask.shape[0]):
            item_pc[point_index] = list(total_point_cloud[mask[point_index][0]]
                                        [mask[point_index][1]][0])
            # print(item_pc[point_index])
            # input('>>>>>>>>>>>>>>>>>>>>>>>>>')
            item_color[point_index] = list(
                color_img[mask[point_index][0]][mask[point_index][1]])
        # item_both = np.hstack((item_pc, item_color / 255))
        tf_matrix = np.loadtxt(f"tf_{camera_num+1}.txt")
        # print(tf_matrix)
        add_line = np.ones([item_pc.shape[0], 1], dtype=float)
        item_pc_a = np.hstack((item_pc, add_line))
        new_item_pc = np.delete(((tf_matrix.dot(item_pc_a.T)).T), 3, axis=1)

        item_dict[name] = [new_item_pc, item_color]
        # np.save(name + '_pc', item_pc)
        # np.save(name + '_color', item_color)
    print(item_dict)

    return item_dict


def camera_detect(model, serials, cams_cut):
    item_dicts = []

    for camera_num, serial in enumerate(serials):
        # Set the Camera
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
        # Start streaming
        pipeline.start(config)

        align = rs.align(rs.stream.color)
        pc = rs.pointcloud()

        if camera_num == 1:
            for i in range(20):
                # print('Camera heating! Wait for a second!')
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                depth = frames.get_depth_frame()
                color = frames.get_color_frame()
            print('Camera2 heating over.')

        for i in range(1):
            print(f'----------第{i+1}张照片-----------')
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth = frames.get_depth_frame()
            color = frames.get_color_frame()

            img_color = np.asanyarray(color.get_data())
            # img_depth = np.asanyarray(depth.get_data())
            pc.map_to(color)
            points = pc.calculate(depth)
            vtx = np.asanyarray(points.get_vertices())
            vtx = np.reshape(vtx, (720, 1280, -1))

            img_color = img_color[
                cams_cut[camera_num][0][0]:cams_cut[camera_num][0][1],
                cams_cut[camera_num][1][0]:cams_cut[camera_num][1][1], :]

            vtx = vtx[cams_cut[camera_num][0][0]:cams_cut[camera_num][0][1],
                      cams_cut[camera_num][1][0]:cams_cut[camera_num][1][1], :]

            target_items_dict = detect_items_in_image(img_color, model)
            # 更改img_color的RGB顺序以正常显示
            item_dict = save_items_point_cloud(vtx, img_color[:, :, [2, 1, 0]],
                                               target_items_dict, camera_num)
            item_dicts.append(item_dict)
        # print(points[:3])

    return item_dicts[0], item_dicts[1]


def pc_add(item1_pc, item2_pc):
    item1_medium_point = np.array(DataFrame(item1_pc).median())
    item2_medium_point = np.array(DataFrame(item2_pc).median())
    medium_point = np.array(np.mean([item1_medium_point, item2_medium_point], axis=0))
    print(f"medium_point: {medium_point}")
    print(item1_medium_point, item2_medium_point)
    item1_movement = medium_point - item1_medium_point
    item2_movement = medium_point - item2_medium_point
    print(f"item1_movement:{item1_movement}, item2_movement:{item2_movement}")

    for i, point in enumerate(item1_pc):
        new_point = np.array(point) + item1_movement
        item1_pc[i] = new_point
    for i, point in enumerate(item2_pc):
        new_point = np.array(point) + item2_movement
        item2_pc[i] = new_point

    item_pc_add = np.append(item1_pc, item2_pc, axis=0)

    return item_pc_add


def pc_show(item_pc, item_color):
    '''使用open3d显示融合点云及位姿'''
    pca = PCA(n_components=2)
    pca.fit(item_pc)

    # 判断特征向量效果
    # print(pca.explained_variance_ratio_)

    feature_vector = pca.components_
    fv1 = np.array(feature_vector[0])
    fv2 = -np.array(feature_vector[1])
    fv3 = np.cross(fv1, fv2)
    # print(fv1.dot(fv2.T))

    medium_point = np.array(DataFrame(item_pc).median())

    item_both = np.hstack((item_pc, item_color / 255))
    # 取消numpy默认的科学计数法，测试表明open3d点云读取函数没法读取科学计数法的表示
    np.set_printoptions(suppress=True)
    # 以txt格式保存点云以方便open3d读取
    np.savetxt('temp.txt', item_both)

    # 在原点云中画图，第一个点为图中的坐标原点
    point = [
        medium_point, medium_point + fv1, medium_point + fv2,
        medium_point + fv3
    ]
    # 由点构成线，[0, 1]代表点集中序号为0和1的点组成线，即原点和三个成分向量划线
    lines = [[0, 1], [0, 2], [0, 3]]
    # 为不同的线添加不同的颜色
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # 构造open3d中的 LineSet对象，用于主成分的显示
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    pcd = o3d.io.read_point_cloud('temp.txt', format='xyzrgb')
    o3d.visualization.draw_geometries([pcd, line_set])


def main():
    # input('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # Load model
    model = MaskRCNN()

    ctx = rs.context()

    # 通过程序去获取已连接摄像头序列号
    serial1 = ctx.devices[0].get_info(rs.camera_info.serial_number)
    serial2 = ctx.devices[1].get_info(rs.camera_info.serial_number)
    print(serial1, serial2)

    item = 'croissant'
    # 相机输入图片切割 原始：[[0, 720], [0, 1280]]
    cam1_cut = [[200, 720], [400, 1120]]
    cam2_cut = [[200, 720], [200, 1180]]
    items_cam1, items_cam2 = camera_detect(model, [serial1, serial2],
                                           [cam1_cut, cam2_cut])
    item_pc_add = pc_add(items_cam1[item][0], items_cam2[item][0])
    # item_pc_add = np.append(items_cam1[item][0],
    #                            items_cam2[item][0],
    #                            axis=0)
    item_color_add = np.append(items_cam1[item][1],
                               items_cam2[item][1],
                               axis=0)
    if item_pc_add.any():
        pc_show(item_pc_add, item_color_add)


if __name__ == "__main__":
    main()
