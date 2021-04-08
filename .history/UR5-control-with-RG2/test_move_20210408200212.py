import numpy as np
import test_main as grasp

carrot_dot = np.asarray([0.03987933, 0.10607725, 0.57800001, 1.])

tf = np.array([[-0.93386511, -0.35511406, -0.04230801, 0.03982764], [-0.00973868, 0.14351067, -0.98960085, -0.83160342], [0.35749283, -0.92374168, -0.13747795,  0.83343039], [ 0.          ,0.         ,0.          ,1.        ]], dtype=float)

new_dot = tf.dot(carrot_dot)
current_tcp = grasp.get_current_tcp()
move_tcp = np.hstack((new_dot[:3],current_tcp[3:]))
print(move_tcp)