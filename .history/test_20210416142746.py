'''
import tensorflow as tf
# 检查tensorflow是否得到CUDA支持，安装成功则显示true，否则为false
print(tf.test.is_built_with_cuda())
# 检查tensorflow是否可以获取到GPU，安装成功则显示true，否则为false
print(tf.test.is_gpu_available())
'''
import numpy as np

item = "croissant"
item_pc = np.load(f"{item}_pc.npy")
print (item_pc)
