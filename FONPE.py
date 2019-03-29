import numpy as np

def f(F,W,P,X,beta):#计算函数的目标函数值
    value = 0
    value = np.linalg.norm(F-np.dot(F,W.transpose()),ord = 'fro')+beta*np.linalg.norm(np.dot(P.transpose(),X)-F, ord = 'fro')
    return value

def FONPE(data,d):
    '''本文执行的是FONPE算法，Pang T, Nie F, Han J. Flexible Orthogonal Neighborhood Preserving Embedding[C]//IJCAI. 2017: 2592-2598.
    输入参数: data:输入的数据, 每一行代表一个样本,每一列代表一个特征  d:降维后的样本的维数
    输出: 每一行代表一个样本,每一列代表一个特征'''
    data = np.array(data)
    X = data.transpose()#将data转置为每一列代表一个样本
    P = np.eye(data.shape[1])#P初始化为单位矩阵
    F = X
    N_MAX=100#算法迭代的最大次数
    beta=5#参数beta
    r = 5#r对应于原文中的lambda参数
    value_best = np.inf#目标函数值初始化无穷大
    #保存原先的结果,目的是寻找最优结果
    P_before = P
    W = np.zeros((data.shape[0],data.shape[0]))
    W_before = W
    F_before = F
    for i in range(N_MAX):
        if f(F,W,P,X,beta) < value_best:#当前目标函数值小于之前的最优值
            #第一步更新W
            W = 0.5*(np.dot(r*np.ones((data.shape[0],data.shape[0]),dtype=np.int)+2*np.dot(F.transpose(),F),np.linalg.pinv(np.dot(F.transpose(),F))))
            for m in range(W.shape[0]):#将权值进行归一化处理
                for n in range(W.shape[1]):
                    W[m,n]=W[m,n]/np.sum(W[m,:])
            #第二步更新P
            #计算矩阵B
            B = beta*np.dot(X,np.linalg.inv(np.dot((np.eye(data.shape[0])-W).transpose(),np.eye(data.shape[0])-W)+beta*np.eye(data.shape[0])))
            #计算矩阵Q
            Q = np.dot(B,np.dot((np.eye(data.shape[0])-W).transpose(),np.dot(np.eye(data.shape[0])-W,B.transpose()))) + beta*(np.dot(X,X.transpose()) - np.dot(X,B.transpose()) - np.dot(B,X.transpose()) + np.dot(B,B.transpose()))
            #对Q进行求特征值和特征向量
            feature_values, feature_vectors = np.linalg.eig(Q)
            tempvalues = np.argsort(feature_values)#对特征值进行排序,返回索引号
            P = np.zeros((data.shape[1],d))#初始化P
            for j in range(1,d+1):#获取从第2小到第d+1小的特征值所对应的特征向量
                P[:,j-1]=feature_vectors[:,tempvalues[j]]
            #第三步更新F
            F = np.dot(P.transpose(),B)
            if f(F,W,P,X,beta) < value_best:#迭代的误差进一步降低,进行下一次迭代
                P_before = P
                W_before = W
                F_before = F
            else:#迭代的误差不降低,终止迭代过程
                P = P_before
                W = W_before
                F = F_before
                break
        else:
            break
    Y = np.dot(P.transpose(),X)#Y矩阵即是最终的降维结果
    Y = Y.transpose()#把降维结果变成与输入结果保持形式一致,即每一行代表一个样本,每一列代表一个特征
    return Y