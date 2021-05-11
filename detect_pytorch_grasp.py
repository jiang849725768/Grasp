import time
import numpy as np
import pyrealsense2 as rs
from pandas import DataFrame

# import os
import sys
# 防止ros中python2的opencv干扰导入
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append('/home/jiang/Grasp/ur5_grasp')
import cv2
import open3d as o3d

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# 引入以下注释
from detectron2.data import MetadataCatalog

from detectron2.utils.visualizer import ColorMode
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import test_main as grasp

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

    # 取消numpy默认的科学计数法，测试表明open3d点云读取函数没法读取科学计数法的表示
    np.set_printoptions(suppress=True)
    fv1 = np.array(feature_vector[0])
    if fv1[2] > 0:
        fv1 = -1 * fv1
    fv2 = np.array(feature_vector[1])
    fv3 = -np.cross(fv1, fv2)

    # print(fv1.dot(fv2.T))
    medium_point = np.array(DataFrame(item_pc).median())
    item_both = np.hstack((item_pc, item_color / 255))

    np.savetxt('temp.txt', item_both)

    # 在原点云中画图
    point = [
        medium_point, medium_point + fv1, medium_point + fv2,
        medium_point + fv3
    ]
    # 第一个点为图中的坐标原点
    lines = [[0, 1], [0, 2], [0, 3]]
    # 由点构成线，[0, 1]代表点集中序号为0和1的点组成线，即原点和三个成分向量划线
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 为不同的线添加不同的颜色(RGB/255)
    # 构造open3d中的 LineSet对象，用于主成分的显示
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    pcd = o3d.io.read_point_cloud('temp.txt', format='xyzrgb')
    o3d.visualization.draw_geometries([pcd, line_set])

    return medium_point, np.array([fv1, fv2, fv3])


def save_items_point_cloud(total_point_cloud, color_img, target_items_dict,
                           cam_num):
    item_dict = {}
    for name, value in target_items_dict.items():
        mask = value[-1]
        item_pc = np.zeros((mask.shape[0], 3), dtype=float)
        item_color = np.zeros((mask.shape[0], 3), dtype=np.uint8)
        for point_index in range(mask.shape[0]):
            item_pc[point_index] = list(total_point_cloud[mask[point_index][0]]
                                        [mask[point_index][1]][0])
            item_color[point_index] = list(
                color_img[mask[point_index][0]][mask[point_index][1]])
        tf_matrix = np.loadtxt(f"tf_{cam_num+1}.txt")

        add_line = np.ones([item_pc.shape[0], 1], dtype=float)
        item_pc_a = np.hstack((item_pc, add_line))
        new_item_pc = np.delete(((tf_matrix.dot(item_pc_a.T)).T), 3, axis=1)
        medium_point, feature_vector = line_set(new_item_pc, item_color)
        rotation_matrix = feature_vector[[2, 1, 0], :].T
        # print(rotation_matrix)
        # medium_point = np.append(medium_point, 1.0)
        # real_point = tf_matrix.dot(medium_point)[:3]
        rotation_vector = cv2.Rodrigues(rotation_matrix)[0]
        tcp = np.hstack((medium_point, rotation_vector.T[0]))
        print(tcp)
        item_dict[name] = [medium_point, feature_vector, rotation_vector, tcp]

    print(item_dict)

    return item_dict


def detect_items_in_image(img, predictor, item_metadata):
    target_items_dict = {}

    s_time = time.time()
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
    e_time = time.time()
    print('detectron2 using : ',
          int(round(e_time * 1000)) - int(round(s_time * 1000)))

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


def camera_detect(predictor, serials, item_metadata):
    items_dicts = []

    # 相机图片切取
    cams_cut = [[[300, 650], [240, 880]], [[140, 720], [90, 980]]]

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

        for i in range(20):
            # print('Camera heating! Wait for a second!')
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
        print('Camera heating over.')

        print(f'----------{camera_num+1}号相机-----------')
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

        cam_cut = cams_cut[camera_num]
        img_cut = img_color[cam_cut[0][0]:cam_cut[0][1],
                            cam_cut[1][0]:cam_cut[1][1], :]
        vtx = vtx[cam_cut[0][0]:cam_cut[0][1], cam_cut[1][0]:cam_cut[1][1], :]
        target_items_dict = detect_items_in_image(img_cut, predictor,
                                                  item_metadata)
        items_dict = save_items_point_cloud(vtx, img_cut[:, :, [2, 1, 0]],
                                            target_items_dict, camera_num)
        items_dicts.append(items_dict)
        # print(points[:3])

    print(f"lenth:{len(items_dicts)}")

    return items_dicts


def item_grasp(tcp, vector_z):
    '''物体抓取'''

    vector_z = 0.01 * vector_z
    print(tcp)
    grasp.operate_gripper(100)
    grasp.move_to_tcp(tcp)
    grasp.increase_move(vector_z[0], vector_z[1], vector_z[2])
    grasp.grasp()
    grasp.move_to_home()


def main(args, item_metadata):
    # input('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # Load model
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)

    ctx = rs.context()
    serials = []
    for i in range(len(ctx.devices)):
        serial_num = ctx.devices[i].get_info(rs.camera_info.serial_number)
        serials.append(serial_num)
        print(f"serial{i}:{serial_num}")

    target_items = ['baozi', 'detergent']
    for item in target_items:
        items_dicts = camera_detect(predictor, serials, item_metadata)
        items_cam1, items_cam2 = items_dicts[0], items_dicts[1]
        if item in items_cam1.keys() and item in items_cam2.keys():
            medium_point = np.mean([items_cam1[item][0], items_cam2[item][0]],
                                   axis=0)
            feature_vector = []
            for i in range(3):
                feature_vector.append(
                    np.mean([items_cam1[item][1][i], items_cam2[item][1][i]],
                            axis=0))
            print(medium_point, feature_vector)
            rotation_matrix = np.array(feature_vector)[[2, 1, 0], :].T
            rotation_vector = cv2.Rodrigues(rotation_matrix)[0]
            tcp = np.hstack((medium_point, rotation_vector.T[0]))
            # print(tcp)

            item_grasp(tcp, feature_vector[0])

    # [ 0.11446287 -0.63587084  0.01617716 -1.5875306  -1.85323039 -0.52081385]

    # if item in items_cam2.keys():
    #     tcp = items_cam2[item][-1]
    #     vector_z = items_cam2[item][1][0]
    #     item_grasp(tcp, vector_z)


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
