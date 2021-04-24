import os
import sys
# import cv2
import colorsys
import numpy as np
import pyrealsense2 as rs
import time
import matplotlib.pyplot as plt

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
    main()
