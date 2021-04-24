import pyrealsense2 as rs
import numpy as np
import cv2
import time

import threading

# from worker import get_worker, get_ptx
from realsense_device_manager_old import DeviceManager

CAM_ORDER = ['', '']

ori_w = 1280
ori_h = 720
fps = 15

det_w = 512
det_h = 320  # x%64==0
det_ratio_w = det_w / float(ori_w)
det_ratio_h = det_h / float(ori_h)
det_ratio = (det_ratio_w, det_ratio_h)

show_h = 288
show_w = 512


def main(debug, noangle):
    if noangle:
        SETCONFIGS = {
            '': None,
            '': None,
        }
    else:
        SETCONFIGS = {
            '': ('mid', 45),
            '': {'left', 45},
        }

    ## test save data
    test_data = []
    average_data = []

    # Configure depth and color streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, ori_w, ori_h, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, ori_w, ori_h, rs.format.bgr8, fps)

    # Use the device manager class to enable the devices and get the frames
    device_manager = DeviceManager(rs.context(), config)
    device_manager.enable_all_devices()
    frames_dict_q = []

    def get_frames_process():
        tmp = device_manager.wait_frames()
        frames_dict_q.append(tmp)

    frames_dict_q.append(device_manager.wait_frames())
    try:
        cnt = 0
        while True:
            if cnt == 0:
                ts = time.time()

            ###### batch to get boxes
            imgs = []
            ks = []
            box_dict = {}
            frames_dict = frames_dict_q.pop(0)
            for k, v in frames_dict.items():
                depth_image, color_image = v
                img = cv2.resize(color_image, \
                        (int(det_w), int(det_h)), \
                        interpolation=cv2.INTER_AREA)
                imgs.append(img)
                ks.append(k)

            # start process for get frames
            t = threading.Thread(target=get_frames_process)
            t.start()

            ###### get depth after boxes
            imgs = {}
            txts = []
            is_warns = 0
            for k, v in frames_dict.items():
                depth_image, color_image = v

                img = cv2.resize(color_image, (show_w,show_h), \
                    interpolation=cv2.INTER_AREA)
                imgs[k] = img

            # Show images
            if len(imgs) == 3:
                img1row = np.hstack((imgs[CAM_ORDER[0]], imgs[CAM_ORDER[2]]))
                img2row = np.hstack((imgs[CAM_ORDER[1]],
                                     np.zeros([show_h, show_w, 3],
                                              dtype=np.uint8)))
                tot_img = np.concatenate([img1row, img2row])
            else:
                tot_img = np.concatenate([v for k, v in imgs.items()])
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', tot_img)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        device_manager.disable_streams()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-a', '--noangle', action='store_true', default=False)
    args = parser.parse_args()
    main(args.debug, args.noangle)