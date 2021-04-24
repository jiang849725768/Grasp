# -*- coding: utf-8 -*-
"""
@File    : test-191204-两个摄像头调用多线程识别.py
@Time    : 2019/12/3 14:12
@Author  : Dontla
@Email   : sxana@qq.com
@Software: PyCharm
"""
# import threading
import cv2
import numpy as np
import pyrealsense2 as rs
# import core.utils as utils
# from core.config import cfg
# from core.yolov3 import YOLOV3
# import tensorflow as tf
# import dontla_package.dontla_ThreadClass as dt

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
    config1.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config2.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline1.start(config1)
    pipeline2.start(config2)
    print("Start successful")

    # 创建对齐对象（深度对齐颜色）
    align1 = rs.align(rs.stream.color)
    align2 = rs.align(rs.stream.color)

    try:
        for i in range(10):
            # while True:
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
            # if not aligned_depth_frame1 or not color_frame1:
            #     continue
            # if not aligned_depth_frame2 or not color_frame2:
            #     continue
            color_image1 = np.asanyarray(color_frame1.get_data())
            color_image2 = np.asanyarray(color_frame2.get_data())
            print(color_image1.shape)
            print(color_image2.shape)

            cv2.imshow("image1.jpg", color_image1)
            cv2.imwrite("image2.jpg", color_image2)
            cv2.waitKey(1)

            # rgb_image1 = color_image1[:, :, [2, 1, 0]]
            # rgb_image2 = color_image2[:, :, [2, 1, 0]]

            # plt.imshow(rgb_image1)
            # plt.imshow(rgb_image2)

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


def main():
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

    for i in range(5):
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
        # rgb_image[:, :, [0, 2]] = rgb_image[:, :, [2, 0]]
        # print(rgb_image.shape)
        plt.imshow(rgb_image)
        cv2.imwrite("0.jpg", rgb_image)
        # plt.pause(1)  # pause 1 second
        # plt.clf()
        # target_objects_dict = detect_objects_in_image(img_color, model)

        # save_objects_point_cloud(vtx, rgb_image, target_objects_dict)

    # print(points[:3])


if __name__ == '__main__':
    dontla_evaluate_detect()
    # main()
