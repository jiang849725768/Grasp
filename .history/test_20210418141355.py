'''
import tensorflow as tf
# 检查tensorflow是否得到CUDA支持，安装成功则显示true，否则为false
print(tf.test.is_built_with_cuda())
# 检查tensorflow是否可以获取到GPU，安装成功则显示true，否则为false
print(tf.test.is_gpu_available())
'''

# -*- coding: UTF-8 -*-
import sys
# 防止ros中python2的opencv干扰导入
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
# a = (0.2,0.4,0.8)
# print (a)
# R = cv2.Rodrigues(a)
# print (R[0])
# v3 = (R[0][2,1],R[0][0,2],R[0][1,0])
# print (v3)
# c = cv2.Rodrigues(v3)
# print (c[0])
# b = cv2.Rodrigues(R[0])
# print (b[0])
# p = (-2.100418,-2.167796,0.27330)
# print(cv2.Rodrigues(p)[0])

rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
rotation_vector, j = cv2.Rodrigues(rotation_matrix)
print(rotation_vector.T)

new_rotaion_matrix, j = cv2.Rodrigues(rotation_vector.T)
print(new_rotaion_matrix)