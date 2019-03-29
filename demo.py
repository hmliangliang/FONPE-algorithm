import numpy as np
import pandas
import time
import scipy.io as sci
import math
import sklearn.preprocessing as pre
from sklearn import neighbors
import FONPE

if  __name__=='__main__':
    start = time.time()
    data = sci.loadmat('Breastw.mat')
    data = data['Breastw']
    data = np.array(data)
    col = data.shape[1]
    label = data[:, col - 1]
    label = label.reshape((data.shape[0], 1))
    data = data[:, 0:col - 1]
    data = pre.minmax_scale(data)
    mydata = data
    d = 8
    num = int(2 / 3 * data.shape[0])
    if d <= d:  # 注意d必须要满足d<=col
        data = FONPE.FONPE(data, d)  # 注意d必须要满足d<=col
        model = neighbors.KNeighborsClassifier(n_neighbors=5)
        model.fit(data[0:num, :], label[0:num, 0])
        y = model.predict(data[num:data.shape[0], :])
        print('经过降维后算法测试的准确率为:', sum(y == label[num:data.shape[0], 0]) / (data.shape[0] - num))
        model0 = neighbors.KNeighborsClassifier(n_neighbors=5)
        model0.fit(mydata[0:num, :], label[0:num, 0])
        y0 = model0.predict(mydata[num:data.shape[0], :])
        print('未经降维算法测试的准确率为:', sum(y0 == label[num:data.shape[0], 0]) / (data.shape[0] - num))
        end = time.time()
        print('此算法在该数据集上的运行时间为:', end - start)
    else:
        print('输入的降维后的维数不得大于原来数据的维数！')