import colorsys
import time
import random
import numpy as np
import pyrealsense2 as rs
from pandas import DataFrame

import os
import sys
# 防止ros中python2的opencv干扰导入
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# 引入以下注释
from detectron2.data import MetadataCatalog

from detectron2.utils.visualizer import ColorMode
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# import random
# import matplotlib.pyplot as plt
# import open3d as o3d

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ROOT_DIR = os.path.abspath("/home/jiang/Grasp")

CLASS_NAMES = [
    "BG", "red_pepper", 'green_pepper', "carrot", "turnip", "eggplant",
    "baozi", "croissant", "cupcake", "ginger", "cake", "corn", "grape",
    "banana", "kiwi", "lemon", "pear", "apple", "carambola", "train",
    "detergent", "plate_w", "plate_g", "paper_box", "plastic_box", "cup",
    "mouse", "hand", "watermelon"
]


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # args.config_file = "../configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )  # 从config file 覆盖配置
    # cfg.merge_from_list(args.opts)  # 从CLI参数 覆盖配置

    # cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'
    # # 本句一定要看下注释！！！！！！！！
    cfg.MODEL.RETINANET.NUM_CLASSES = 28
    # # 类别数+1（因为有background，也就是你的 cate id 从 1 开始，如果您的数据集Json下标从 0 开始，这个改为您对应的类别就行，不用再加背景类！！！！！）
    # # cfg.MODEL.WEIGHTS="/home/yourstorePath/.pth"
    cfg.MODEL.WEIGHTS = "./model_final.pth"  # 已有模型权重
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


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
    visualize.display_instances(save_path_name,
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
                if mask[x][y]:
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
        object_dict[name] = [medium_point, feature_vector, object_both]
        np.save(name + '_pc', object_pc)
        np.save(name + '_color', object_color)
    print(object_dict)

    # return object_both


def detect_items_in_image(img, args, item_metadata):
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    # format is documented at
    # https://detectron2.readthedocs.io/tutorials/models.html # model-output-format
    v = Visualizer(
        img[:, :, ::-1],
        metadata=item_metadata,
        scale=0.5,
        instance_mode=ColorMode.IMAGE_BW
        # remove the colors of unsegmented pixels.
        # This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(instances)
    masks_arr = instances.pred_masks.numpy()
    item_classes = instances.pred_classes.numpy()
    if item_classes.any():
        print(item_classes)
    print(masks_arr.shape)
    if masks_arr.shape[-1] == 1:
        print("oops")
    arr_h, arr_w, _ = np.nonzero(masks_arr)
    print(f"arr_h:{arr_h};arr_w:{arr_w}")

    plt.imshow(out.get_image())
    plt.show()


def main(args, item_metadata):
    # input('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # Load model
    # model = MaskRCNN()

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
        # img_depth = np.asanyarray(depth.get_data())
        pc.map_to(color)
        points = pc.calculate(depth)
        vtx = np.asanyarray(points.get_vertices())
        vtx = np.reshape(vtx, (720, 1280, -1))
        cam_cut = [[200, 720], [200, 1180]]

        # get 3d point cloud (masked)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        rgb_image = np.asanyarray(color_frame.get_data())
        rgb_image[:, :, [0, 2]] = rgb_image[:, :, [2, 0]]
        img_cut = img_color[cam_cut[0][0]:cam_cut[0][1],
                            cam_cut[1][0]:cam_cut[1][1], :]
        # rgb_image_cut = rgb_image[cam_cut[0][0]:cam_cut[0][1], cam_cut[1][0]:cam_cut[1][1], :]
        detect_items_in_image(img_cut, args, item_metadata)
        # print(rgb_image.shape)
        # plt.imshow(rgb_image)
        # plt.pause(1)  # pause 1 second
        # plt.clf()
        # target_objects_dict = detect_objects_in_image(img_color, model)

        # save_objects_point_cloud(vtx, rgb_image, target_objects_dict)

    # print(points[:3])


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    item_metadata = MetadataCatalog.get("coco_my_train").set(
        thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
        evaluator_type='coco',  # 指定评估方式
    )
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(
            args,
            item_metadata,
        ),
    )
    main()
