import sys
# 防止ros中python2的opencv干扰导入
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

# import matplotlib.pyplot as plt  # plt 用于显示图片
# import matplotlib.image as mpimg  # mpimg 用于读取图片

import os
import numpy as np
import glob


class shuangmu():
    def __init__(self):
        self.m1 = 0
        self.m2 = 0
        self.d1 = 0
        self.d2 = 0
        self.R = 0
        self.T = 0


stereo = shuangmu()


class StereoCalibration(object):
    def __init__(self):
        self.imagesL = self.read_images('d455')
        self.imagesR = self.read_images('d435')

    def read_images(self, cal_path):
        filepath = glob.glob(
            os.path.join('calibration_test', cal_path, '*.png'))
        print(os.path.join(cal_path, '*.png'))
        print(filepath)
        filepath.sort()
        return filepath

    # 标定图像
    def calibration_photo(self):
        # 设置要标定的角点个数
        x_nums = 5  # x方向上的角点个数
        y_nums = 8
        # 设置(生成)标定图在世界坐标中的坐标
        world_point = np.zeros(
            (x_nums * y_nums, 3),
            np.float32)  # 生成x_nums*y_nums个坐标，每个坐标包含x,y,z三个元素
        world_point[:, :2] = np.mgrid[:x_nums, :y_nums].T.reshape(-1, 2)
        # mgrid[]生成包含两个二维矩阵的矩阵，每个矩阵都有x_nums列,y_nums行
        # .T矩阵的转置
        # reshape()重新规划矩阵，但不改变矩阵元素
        # 保存角点坐标
        world_position = []
        image_positionl = []
        image_positionr = []
        # 设置角点查找限制
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,
                    0.001)
        # 获取所有标定图
        for ii in range(1):

            image_path_l = self.imagesL[ii]
            image_path_r = self.imagesR[ii]

            image_l = cv2.imread(image_path_l)
            image_r = cv2.imread(image_path_r)
            gray_l = cv2.cvtColor(image_l, cv2.COLOR_RGB2GRAY)
            gray_r = cv2.cvtColor(image_r, cv2.COLOR_RGB2GRAY)

            # 查找角点
            #         ok,corners = cv2.findChessboardCorners(gray,(x_nums,y_nums),None)
            ok1, cornersl = cv2.findChessboardCorners(gray_l, (x_nums, y_nums),
                                                      None)
            ok2, cornersr = cv2.findChessboardCorners(gray_r, (x_nums, y_nums),
                                                      None)
            # ok1, cornersl = cv2.findCirclesGrid(gray_l, (x_nums, y_nums), None)
            # ok2, cornersr = cv2.findCirclesGrid(gray_r, (x_nums, y_nums), None)

            self.world = world_point
            print(ok1)
            print(ok2)
            if ok1 & ok2:
                # 把每一幅图像的世界坐标放到world_position中
                center_spacing = 15  # 圆心的位置距离，这一个其实不重要
                world_position.append(world_point * center_spacing)
                # 获取更精确的角点位置
                exact_cornersl = cv2.cornerSubPix(gray_l, cornersl, (11, 11),
                                                  (-1, -1), criteria)
                exact_cornersr = cv2.cornerSubPix(gray_r, cornersr, (11, 11),
                                                  (-1, -1), criteria)
                # 把获取的角点坐标放到image_position中
                image_positionl.append(exact_cornersl)
                image_positionr.append(exact_cornersr)
                # 可视化角点

                # image = cv2.drawChessboardCorners(image, (x_nums, y_nums),
                #                                   exact_corners, ok)
                # cv2.imshow('image_corner', image)
                # cv2.waitKey(0)

    # 计算内参数
        image_shape = gray_l.shape[::-1]

        retl, mtxl, distl, rvecsl, tvecsl = cv2.calibrateCamera(
            world_position, image_positionl, image_shape, None, None)
        retr, mtxr, distr, rvecsr, tvecsr = cv2.calibrateCamera(
            world_position, image_positionr, image_shape, None, None)
        print('ml = ', mtxl)
        print('mr = ', mtxr)
        print('dl = ', distl)
        print('dr = ', distr)
        stereo.m1 = mtxl
        stereo.m2 = mtxr
        stereo.d1 = distl
        stereo.d2 = distr

        # 计算误差
        self.cal_error(world_position, image_positionl, mtxl, distl, rvecsl,
                       tvecsl)
        self.cal_error(world_position, image_positionr, mtxr, distr, rvecsr,
                       tvecsr)

        # 双目标定
        self.stereo_calibrate(world_position, image_positionl, image_positionr,
                              mtxl, distl, mtxr, distr, image_shape)

    def cal_error(self, world_position, image_position, mtx, dist, rvecs,
                  tvecs):
        # 计算偏差
        mean_error = 0
        for i in range(len(world_position)):
            image_position2, _ = cv2.projectPoints(world_position[i], rvecs[i],
                                                   tvecs[i], mtx, dist)
            error = cv2.norm(image_position[i], image_position2,
                             cv2.NORM_L2) / len(image_position2)
            mean_error += error
        print("total error: ", mean_error / len(image_position))

    def stereo_calibrate(self, objpoints, imgpoints_l, imgpoints_r, M1, d1, M2,
                         d2, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_l,
            imgpoints_r,
            M1,
            d1,
            M2,
            d2,
            dims,
            criteria=stereocalib_criteria,
            flags=flags)
        print(R)
        print(T)
        stereo.R = R
        stereo.T = T


if __name__ == '__main__':
    #     calibration_photo()
    biaoding = StereoCalibration()
    biaoding.calibration_photo()
