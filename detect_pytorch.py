import numpy as np
import pyrealsense2 as rs
from pandas import DataFrame

import os
import sys
# 防止ros中python2的opencv干扰导入
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# import colorsys, time, random, cv2

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

# ROOT_DIR = os.path.abspath("/home/jiang/Grasp")

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
    cfg.MODEL.WEIGHTS = "./model_final_12.pth"  # 已有模型权重
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


__k = 0


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


def save_objects_point_cloud(
    total_point_cloud,
    color_img,
    target_objects_dict,
):
    # np.save('full_point_cloud_test', total_point_cloud)
    # np.save('full_color', color_img)
    object_dict = {}
    for name, value in target_objects_dict.items():
        mask = value[-1]
        object_pc = np.zeros((mask.shape[0], 3), dtype=float)
        object_color = np.zeros((mask.shape[0], 3), dtype=np.uint8)
        for point_index in range(mask.shape[0]):
            # object_pc[point_index] = list(
            #     total_point_cloud[mask[point_index][0] +
            #                       cam_cut[0][0]][mask[point_index][1] +
            #                                      cam_cut[1][0]][0])
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
    print(masks_arr.shape)
    target_items_dict = {}
    # for i in range(masks_arr.shape[0]):
    #     arr_h, arr_w = np.nonzero(masks_arr[i])
    # print(f"arr_h:{arr_h};arr_w:{arr_w}")
    if item_classes.any():
        for i, item_num in enumerate(item_classes):
            print(CLASS_NAMES[item_num])
            arr_h, arr_w = np.nonzero(masks_arr[i])
            mask_points = np.array([arr_h, arr_w]).T
            print(f"mask_points:{mask_points.shape}")
            target_items_dict[CLASS_NAMES[item_num]] = [mask_points]

    res_img = out.get_image()
    plt.imshow(res_img)
    plt.savefig("result/0.jpg")
    plt.show()

    return target_items_dict


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
        cam_cut = [[330, 700], [200, 980]]

        # get 3d point cloud (masked)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        rgb_image = np.asanyarray(color_frame.get_data())
        rgb_image[:, :, [0, 2]] = rgb_image[:, :, [2, 0]]
        img_cut = img_color[cam_cut[0][0]:cam_cut[0][1],
                            cam_cut[1][0]:cam_cut[1][1], :]
        vtx = vtx[cam_cut[0][0]:cam_cut[0][1], cam_cut[1][0]:cam_cut[1][1], :]
        # rgb_image_cut = rgb_image[cam_cut[0][0]:cam_cut[0][1], cam_cut[1][0]:cam_cut[1][1], :]
        target_items_dict = detect_items_in_image(img_cut, args, item_metadata)
        save_objects_point_cloud(vtx, img_cut[:, :, [2, 1, 0]],
                                 target_items_dict)
        # print(rgb_image.shape)
        # plt.imshow(rgb_image)
        # plt.pause(1)  # pause 1 second
        # plt.clf()

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
