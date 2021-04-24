# -*- coding: utf-8 -*-
"""
@File    : test-191204-两个摄像头调用多线程识别.py
@Time    : 2019/12/3 14:12
@Author  : Dontla
@Email   : sxana@qq.com
@Software: PyCharm
"""
import threading

import cv2
import numpy as np
import pyrealsense2 as rs
import time
import matplotlib.pyplot as plt
# import core.utils as utils
# from core.config import cfg
# from core.yolov3 import YOLOV3
# import tensorflow as tf
# import dontla_package.dontla_ThreadClass as dt

'''
class YoloTest(object):

    def __init__(self):

        # D·C 191111：__C.TEST.INPUT_SIZE = 544
        self.input_size = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        # Dontla 191106注释：初始化class.names文件的字典信息属性
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        # D·C 191115：类数量属性
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        # D·C 191111：__C.TEST.SCORE_THRESHOLD = 0.3
        self.score_threshold = cfg.TEST.SCORE_THRESHOLD
        # D·C 191120：__C.TEST.IOU_THRESHOLD = 0.45
        self.iou_threshold = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        # D·C 191120：__C.TEST.ANNOT_PATH = "./data/dataset/Dontla/20191023_Artificial_Flower/test.txt"
        self.annotation_path = cfg.TEST.ANNOT_PATH
        # D·C 191120：__C.TEST.WEIGHT_FILE = "./checkpoint/f_g_c_weights_files/yolov3_test_loss=15.8845.ckpt-47"
        self.weight_file = cfg.TEST.WEIGHT_FILE
        # D·C 191115：可写标记（bool类型值）
        self.write_image = cfg.TEST.WRITE_IMAGE
        # D·C 191115：__C.TEST.WRITE_IMAGE_PATH = "./data/detection/"（识别图片画框并标注文本后写入的图片路径）
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        # D·C 191116：TEST.SHOW_LABEL设置为True
        self.show_label = cfg.TEST.SHOW_LABEL

        # D·C 191120：创建命名空间“input”
        with tf.name_scope('input'):
            # D·C 191120：建立变量（创建占位符开辟内存空间）
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable = tf.placeholder(dtype=tf.bool, name='trainable')

        model = YOLOV3(self.input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox

        # D·C 191120：创建命名空间“指数滑动平均”
        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        # D·C 191120：在允许软设备放置的会话中启动图形并记录放置决策。（不懂啥意思。。。）allow_soft_placement=True表示允许tf自动选择可用的GPU和CPU
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # D·C 191120：variables_to_restore()用于加载模型计算滑动平均值时将影子变量直接映射到变量本身
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        # D·C 191120：用于下次训练时恢复模型
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):
        # D·C 191107：复制一份图片的镜像，避免对图片直接操作改变图片的内在属性
        org_image = np.copy(image)
        # D·C 191107：获取图片尺寸
        org_h, org_w, _ = org_image.shape

        # D·C 191108：该函数将源图结合input_size，将其转换成预投喂的方形图像（作者默认544×544，中间为缩小尺寸的源图，上下空区域为灰图）：
        image_data = utils.image_preprocess(image, [self.input_size, self.input_size])

        # D·C 191108：打印维度看看：
        # print(image_data.shape)
        # (544, 544, 3)

        # D·C 191108：创建新轴，不懂要创建新轴干嘛？
        image_data = image_data[np.newaxis, ...]

        # D·C 191108：打印维度看看：
        # print(image_data.shape)
        # (1, 544, 544, 3)

        # D·C 191110：三个box可能存放了预测框图（可能是N多的框，有用的没用的重叠的都在里面）的信息（但是打印出来的值完全看不懂啊喂？）
        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            }
        )

        # D·C 191110：打印三个box的类型、形状和值看看：
        # print(type(pred_sbbox))
        # print(type(pred_mbbox))
        # print(type(pred_lbbox))
        # 都是<class 'numpy.ndarray'>

        # print(pred_sbbox.shape)
        # print(pred_mbbox.shape)
        # print(pred_lbbox.shape)
        # (1, 68, 68, 3, 6)
        # (1, 34, 34, 3, 6)
        # (1, 17, 17, 3, 6)

        # print(pred_sbbox)
        # print(pred_mbbox)
        # print(pred_lbbox)

        # D·C 191110：（-1，6）表示不知道有多少行，反正你给我整成6列，然后concatenate又把它们仨给叠起来，最终得到无数个6列数组（后面self.num_classes)个数存放的貌似是这个框属于类的概率）
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        # D·C 191111：打印pred_bbox和它的维度看看：
        # print(pred_bbox)
        # print(pred_bbox.shape)
        # (18207, 6)

        # D·C 191111：猜测是第一道过滤，过滤掉score_threshold以下的图片，过滤完之后少了好多：
        # D·C 191115：bboxes维度为[n,6]，前四列是坐标，第五列是得分，第六列是对应类下标
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        # D·C 191111：猜测是第二道过滤，过滤掉iou_threshold以下的图片：
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes
'''

'''
def dontla_evaluate_detect():
    pipeline1 = rs.pipeline()
    pipeline2 = rs.pipeline()
    config1 = rs.config()
    config2 = rs.config()

    ctx = rs.context()

    # 通过程序去获取已连接摄像头序列号
    serial1 = ctx.devices[0].get_info(rs.camera_info.serial_number)
    serial2 = ctx.devices[1].get_info(rs.camera_info.serial_number)
    print(serial1, serial2)

    config1.enable_device(serial1)
    config2.enable_device(serial2)
    config1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline1.start(config1)
    pipeline2.start(config2)

    # 创建对齐对象（深度对齐颜色）
    align1 = rs.align(rs.stream.color)
    align2 = rs.align(rs.stream.color)

    while True:
        frames1 = pipeline1.wait_for_frames()
        frames2 = pipeline2.wait_for_frames()
        # 获取对齐帧集
        aligned_frames1 = align1.process(frames1)
        aligned_frames2 = align2.process(frames2)
        # 获取对齐后的深度帧和彩色帧
        aligned_depth_frame1 = aligned_frames1.get_depth_frame()
        aligned_depth_frame2 = aligned_frames2.get_depth_frame()
        color_frame1 = aligned_frames1.get_color_frame()
        color_frame2 = aligned_frames2.get_color_frame()
        # 获取颜色帧内参
        color_profile1 = color_frame1.get_profile()
        color_profile2 = color_frame2.get_profile()
        cvsprofile1 = rs.video_stream_profile(color_profile1)
        cvsprofile2 = rs.video_stream_profile(color_profile2)
        color_intrin1 = cvsprofile1.get_intrinsics()
        color_intrin2 = cvsprofile2.get_intrinsics()
        color_intrin_part1 = [color_intrin1.ppx, color_intrin1.ppy, color_intrin1.fx, color_intrin1.fy]
        color_intrin_part2 = [color_intrin2.ppx, color_intrin2.ppy, color_intrin2.fx, color_intrin2.fy]

        # if not aligned_depth_frame1 or not color_frame1:
        #     continue
        # if not aligned_depth_frame2 or not color_frame2:
        #     continue
        color_image1 = np.asanyarray(color_frame1.get_data())
        color_image2 = np.asanyarray(color_frame2.get_data())
        # print(color_image1.shape)
        # print(color_image2.shape)

        rgb_image1 = color_image1[:, :, [2, 1, 0]]
        rgb_image2 = color_image2[:, :, [2, 1, 0]]

        plt.imshow(rgb_image1)
        plt.imshow(rgb_image2)


    try:
        while True:
            frames1 = pipeline1.wait_for_frames()
            frames2 = pipeline2.wait_for_frames()
            # 获取对齐帧集
            aligned_frames1 = align1.process(frames1)
            aligned_frames2 = align2.process(frames2)
            # 获取对齐后的深度帧和彩色帧
            aligned_depth_frame1 = aligned_frames1.get_depth_frame()
            aligned_depth_frame2 = aligned_frames2.get_depth_frame()
            color_frame1 = aligned_frames1.get_color_frame()
            color_frame2 = aligned_frames2.get_color_frame()
            # 获取颜色帧内参
            color_profile1 = color_frame1.get_profile()
            color_profile2 = color_frame2.get_profile()
            cvsprofile1 = rs.video_stream_profile(color_profile1)
            cvsprofile2 = rs.video_stream_profile(color_profile2)
            color_intrin1 = cvsprofile1.get_intrinsics()
            color_intrin2 = cvsprofile2.get_intrinsics()
            color_intrin_part1 = [color_intrin1.ppx, color_intrin1.ppy, color_intrin1.fx, color_intrin1.fy]
            color_intrin_part2 = [color_intrin2.ppx, color_intrin2.ppy, color_intrin2.fx, color_intrin2.fy]

            # if not aligned_depth_frame1 or not color_frame1:
            #     continue
            # if not aligned_depth_frame2 or not color_frame2:
            #     continue
            color_image1 = np.asanyarray(color_frame1.get_data())
            color_image2 = np.asanyarray(color_frame2.get_data())
            # print(color_image1.shape)
            # print(color_image2.shape)

            rgb_image1 = color_image1[:, :, [2, 1, 0]]
            rgb_image2 = color_image2[:, :, [2, 1, 0]]

            plt.imshow(rgb_image1)
            plt.imshow(rgb_image2)


            # D·C 191121：显示帧看看
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', color_frame1)
            # cv2.waitKey(1)

            # bboxes_pr1 = self.predict(color_image1)
            # bboxes_pr2 = self.predict(color_image2)

            # thread1 = threading.Thread(target=self.predict, name='thread1', args=(color_image1))
            # 使用自定义线程类
            # begin_time = time.time()
            # thread1 = dt.MyThread(self.predict, (color_image1,))
            # thread2 = dt.MyThread(self.predict, (color_image2,))
            # thread1.start()
            # thread2.start()
            # bboxes_pr1 = thread1.get_result()
            # bboxes_pr2 = thread2.get_result()
            # end_time = time.time()
            # t = end_time - begin_time
            # print('耗时：{}秒'.format(t))

            # image1 = utils.draw_bbox(color_image1, bboxes_pr1, aligned_depth_frame1, color_intrin_part1,
            #                             show_label=self.show_label)
            # image2 = utils.draw_bbox(color_image2, bboxes_pr2, aligned_depth_frame2, color_intrin_part2,
            #                             show_label=self.show_label)

            # # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('window1', image1)
            # cv2.imshow('window2', image2)
            # cv2.waitKey(1)

    finally:
        pipeline1.stop()
        pipeline2.stop()
'''

def pipthing():
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

        # get 3d point cloud (masked)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        rgb_image = np.asanyarray(color_frame.get_data())
        rgb_image[:, :, [0, 2]] = rgb_image[:, :, [2, 0]]
        print(rgb_image.shape)
        plt.imshow(rgb_image)
        # plt.pause(1)  # pause 1 second
        # plt.clf()
        # target_objects_dict = detect_objects_in_image(img_color, model)

        # save_objects_point_cloud(vtx, rgb_image, target_objects_dict)

    # print(points[:3])


if __name__ == '__main__':
    # dontla_evaluate_detect()
    pipthing()

