import numpy as np

# Markov Model
'''
T = np.array([[0.4,0.3,0.3],[0.2,0.6,0.2],[0.1,0.1,0.8]]).T
p1 = np.array([1,0,0])
p2 = np.dot(T,p1)
print(p2)
p3 = np.dot(T,p2)
print(p3)
p4 = np.dot(T,p3)
print(p4)
p5 = np.dot(T,p4)
print(p5)
p6 = np.dot(T,p5)
print(p6)
p7 = np.dot(T,p6)
print(p7)
print('--------------------')
T = np.array([[0.4,0.3,0.3],[0.2,0.6,0.2],[0.1,0.1,0.8]])
p1 = np.array([1,0,0])
t = 20
temp = T
for i in range(0,t-1):
    temp = np.dot(temp,T)
pt = np.dot(p1,temp)
print(pt)
print('--------------------')

T = np.array([[0.1,0.4,0.5],[0.4,0,0.6],[0,0.6,0.4]]).T
p1 = np.array([1,0,0])
p2 = np.dot(T,p1)
print(p2)
p3 = np.dot(T,p2)
print(p3)
p4 = np.dot(T,p3)
print(p4)
p5 = np.dot(T,p4)
print(p5)
p6 = np.dot(T,p5)
print(p6)
p7 = np.dot(T,p6)
print(p7)
print('--------------------')
T = np.array([[0.1,0.4,0.5],[0.4,0,0.6],[0,0.6,0.4]])
p1 = np.array([1,0,0])
t = 20
temp = T
for i in range(0,t-1):
    temp = np.dot(temp,T)
pt = np.dot(p1,temp)
print(pt)
'''
T = np.array([[0,0.5,0.5,0],[0.5,0,0,0.5],[0.5,0,0,0.5],[0,0,0,1]]).T
p = np.array([1,0,0,0]).T
t = 20
temp = T
for i in range(0,t-1):
    p = np.dot(T,p)
    print(p)


x = np.array([
    [1],
    [0],
    [0],
    [0]])  # 当前的状态向量
P = np.array([
    [0, 0.5, 0.5, 0],
    [0.5, 0, 0, 0],
    [0.5, 0, 0, 0],
    [0, 0.5, 0.5, 1]
])  # 转移矩阵
i=1
while i < 20:
    x = P.dot(x)  # x=P*x
    i+=1
    print(x.flatten())
