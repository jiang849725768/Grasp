# import tensorflow as tf
# # 检查tensorflow是否得到CUDA支持，安装成功则显示true，否则为false
# # print(tf.test.is_built_with_cuda())
# # 检查tensorflow是否可以获取到GPU，安装成功则显示true，否则为false
# print(tf.test.is_gpu_available())

import numpy as np

a = [[1,2,3], [2,3,4]]
b = [1,3,4]
for i, point in enumerate(a):
    new_point = np.array(point) + np.array(b)
    a[i] = new_point

print(a)