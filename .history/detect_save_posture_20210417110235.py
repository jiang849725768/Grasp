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
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2



# import random
# import matplotlib.pyplot as plt
# import open3d as o3d

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


def detect_objects_in_image(image, model):
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
    masked_image = visualize.display_instances(save_path_name,
                                               image,
                                               r['rois'],
                                               r['masks'],
                                               r['class_ids'],
                                               class_names,
                                               r['scores'],
                                               colors=COLOR)

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
        '''
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
        '''
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
        object_dict[name] = [
            medium_point, feature_vector, object_both
        ]
        np.save(name + '_pc', object_pc)
        np.save(name + '_color', object_color)
    print(object_dict)

    # return object_both


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

    for i in range(100):
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
        # plt.imshow(rgb_image)
        # plt.pause(1)  # pause 1 second
        # plt.clf()
        target_objects_dict = detect_objects_in_image(img_color, model)

        save_objects_point_cloud(vtx, rgb_image, target_objects_dict)

    # print(points[:3])


if __name__ == "__main__":
    main()
