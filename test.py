'''
import tensorflow as tf
# 检查tensorflow是否得到CUDA支持，安装成功则显示true，否则为false
print(tf.test.is_built_with_cuda())
# 检查tensorflow是否可以获取到GPU，安装成功则显示true，否则为false
print(tf.test.is_gpu_available())
'''
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
print(cv2.__version__)
