import random
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import colorsys
from pandas import DataFrame
# import open3d as o3d

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from sklearn.decomposition import PCA

import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

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

    # RESULT_SAVE_PATH = "/home/jiang/Grasp/result"
    # # !!! change the function to choose which object to show in the results
    # save_path_name = os.path.join(RESULT_SAVE_PATH, 'img.jpg')
    # from mrcnn import visualize
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
                if mask[x][y] is not False:
                    mask_points.append([x, y])
        mask_points = np.array(mask_points)
        if mask_points.shape[0] < 1500:
            continue
        target_object_dict[class_names[index]] = [mask_points]
    return target_object_dict


def line_set(item_pc, item_color):
    pca = PCA(n_components=2)
    pca.fit(item_pc)

    # 判断特征向量效果
    # print(pca.explained_variance_ratio_)

    feature_vector = pca.components_

    fv1 = np.array(feature_vector[0])
    fv2 = -np.array(feature_vector[1])
    fv3 = np.cross(fv1, fv2)

    print(fv1.dot(fv2.T))

    medium_point = np.array(DataFrame(item_pc).median())

    return medium_point, np.array([fv1, fv2, fv3])


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
        medium_point, feature_vector = line_set(object_pc, object_color)
        object_dict[name] = [medium_point, feature_vector, object_pc, object_color]
        np.save(name + '_pc', object_pc)
        np.save(name + '_color', object_color)
    # print(object_dict)

    return object_dict


def go_to_dot(item_pc, item_color):

    move_dot = np.append(dot, 1.0)

    new_dot = tf.dot(move_dot)
    tf_matrix = np.loadtxt("tf.txt")
    # print(tf_matrix)
    add_line = np.ones([item_pc.shape[0], 1], dtype=float)
    item_pc = np.hstack((item_pc, add_line))
    new_item_pc = np.delete(((tf_matrix.dot(item_pc.T)).T), 3, axis=1)

    medium_point, feature_vector = line_set(new_item_pc, item_color)
    rotation_matrix = feature_vector[[2, 1, 0], :]
    # medium_point = np.append(medium_point, 1.0)
    # real_point = tf_matrix.dot(medium_point)[:3]
    rotation_vector = rm2rv(rotation_matrix.T)
    np.hstack((medium_point, rotation_matrix))
    # print(f"current_tcp:{current_tcp}")
    # move_tcp = np.hstack((new_dot[:3], current_tcp[3:]))
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

    for i in range(2):
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
        # 改变rgb顺序
        rgb_image[:, :, [0, 2]] = rgb_image[:, :, [2, 0]]
        # print(rgb_image.shape)

        # 显示rgb图片
        # plt.imshow(rgb_image)

        # plt.pause(1)  # pause 1 second
        # plt.clf()
        target_objects_dict = detect_objects_in_image(img_color, model)

        object_dict = save_objects_point_cloud(vtx, rgb_image,
                                               target_objects_dict)

        test_object = ['carrot', 'croissant']
        for item in test_object:
            if item in object_dict:
                item_pc, item_color = object_dict[item][2], object_dict[item][3]
                go_to_dot(item_pc, item_color)
        time.sleep(3)

    grasp.go_home()


if __name__ == "__main__":
    main()