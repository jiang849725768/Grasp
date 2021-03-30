from math import cos, sin, pi, atan2 ,asin
# 欧拉角(x, y, z)转换为四元数(q0, q1, q2, q3)



# x , y , z 单位为角度
x, y, z  = 0 ,0 ,90
x, y, z  = x*pi/180 ,y*pi/180 ,z*pi/180
q0 ,q1 ,q2 ,q3 = 0 ,0 ,0 ,0
q0 = cos(x/2)*cos(y/2)*cos(z/2) + sin(x/2)*sin(y/2)*sin(z/2)
q1 = sin(x/2)*cos(y/2)*cos(z/2) - cos(x/2)*sin(y/2)*sin(z/2)
q2 = cos(x/2)*sin(y/2)*cos(z/2) + sin(x/2)*cos(y/2)*sin(z/2)
q3 = cos(x/2)*cos(y/2)*sin(z/2) - sin(x/2)*sin(y/2)*cos(z/2)
print('欧拉角({0:f}, {1:f}, {2:f})转换为四元数(q0, q1, q2, q3)'.format(x*180/pi, y*180/pi, z*180/pi))    
print("q0 = {0:f}".format(q0))
print("q1 = {0:f}".format(q1))
print("q2 = {0:f}".format(q2))
print("q3 = {0:f}".format(q3))    
print() 

#四元数q=(q0,q1,q2,q3)到欧拉角(x, y, z)
q0 ,q1 ,q2 ,q3 = 0.707 ,0 ,0 ,0.707
x, y, z  = 0 ,0 ,0
x = atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))*180/pi
y = asin(2*(q0*q2 - q1*q3))*180/pi #asin = arcsin
z = atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))*180/pi
print('四元数q=({0:f},{1:f},{2:f},{3:f})到欧拉角(x, y, z))'.format(q0, q1, q2, q3))    
print("x = {0:f}".format(x))
print("y = {0:f}".format(y))
print("z = {0:f}".format(z))   
print() 

#欧拉角(x, y, z)转换为旋转矩阵
x, y, z  = 0 ,0 ,90
x, y, z  = x*pi/180 ,y*pi/180 ,z*pi/180
r11 ,r12 ,r13 = cos(z)*cos(y), cos(z)*sin(y)*sin(x) - sin(z)*cos(x), cos(z)*sin(y)*cos(x) + sin(z)*sin(x)
r21 ,r22 ,r23 = sin(z)*cos(y), sin(z)*sin(y)*sin(x) + cos(z)*cos(x), sin(z)*sin(y)*cos(x) - cos(z)*sin(x)
r31 ,r32 ,r33 = -sin(y), cos(y)*sin(x), cos(y)*cos(x)
print('欧拉角({0:f}, {1:f}, {2:f})转换为旋转矩阵'.format(x*180/pi, y*180/pi, z*180/pi))      
print('{0:f} , {1:f} , {2:f}'.format(r11 ,r12 ,r13))
print('{0:f} , {1:f} , {2:f}'.format(r21 ,r22 ,r23))
print('{0:f} , {1:f} , {2:f}'.format(r31 ,r32 ,r33))    
print() 

#四元数q=(q0,q1,q2,q3)到旋转矩阵
q0 ,q1 ,q2 ,q3 = 0.707 ,0 ,0 ,0.707
r11 ,r12 ,r13 = 1-2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)
r21 ,r22 ,r23 = 2*(q1*q2 + q0*q3), 1-2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)
r31 ,r32 ,r33 = 2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1-2*(q1*q1 + q2*q2)
print('四元数q=({0:f},{1:f},{2:f},{3:f})转换为旋转矩阵'.format(q0, q1, q2, q3))    
print('{0:f} , {1:f} , {2:f}'.format(r11 ,r12 ,r13))
print('{0:f} , {1:f} , {2:f}'.format(r21 ,r22 ,r23))
print('{0:f} , {1:f} , {2:f}'.format(r31 ,r32 ,r33))      
print() 
