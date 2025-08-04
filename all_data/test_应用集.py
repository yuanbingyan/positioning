import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from All_data.adaboost import adaboost_main
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
from All_data.data_op_应用集 import data_op_main
import time
### 使用应用集，将训练集设置为全部带经纬度点的数据。
def rad(d):
    return d * np.pi / 180.0
def distance(lng1, lng2, lat1, lat2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * np.arcsin(np.sqrt(pow(np.sin(a / 2), 2) + np.cos(radLat1) * np.cos(radLat2) * pow(np.sin(b / 2), 2)))
    s = s * 6378.137 * 1000
    return s
def train(): ## 训练模型
    limli_scale_data = pd.read_csv('计算覆盖距离.csv',encoding='ANSI')
    data_with_GPS = pd.read_excel('华为已处理数据.xlsx')
    BS_lng_all,BS_lat_all=[],[]
    for index,row in limli_scale_data.iterrows():
        lte_id = row['lte_id']
        limli_scale = row['覆盖范围']
        # limli_scale=500
        if limli_scale<=500:
            limli_scale = 500
        BS_lng = data_with_GPS[data_with_GPS['lte_id'] == lte_id].经度.iloc[0]
        BS_lat = data_with_GPS[data_with_GPS['lte_id'] == lte_id].纬度.iloc[0]
        BS_lng_all.append(BS_lng)
        BS_lat_all.append(BS_lat)
        lte_id_data = data_with_GPS[data_with_GPS['lte_id'] == lte_id][['ltescrsrp','ltescrsrq','ltescphr','ltencrsrp_1','ltencrsrq_1','ltencpci_1','ltencrsrp_2','ltencrsrq_2','ltencpci_2','longitude','latitude']]
        lte_id_data = np.array(lte_id_data)
        ###
        s = distance(BS_lng, lte_id_data[:,9], BS_lat, lte_id_data[:,10])
        num = []
        for j in range(len(s)):
            if s[j] > limli_scale:
                num.append(j)
        # print('500米范围占',1-len(num)/len(s))
        lte_id_data = np.delete(lte_id_data, num, axis=0)

        ###
        train_x, test_x, train_y, test_y = train_test_split(lte_id_data[:,0:9], lte_id_data[:,9:], test_size=0.0001)
        print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
        lte_id_data = np.concatenate((train_x,train_y),axis=1)
        np.savez('data/{}_应用集.npz'.format(lte_id),lte_id_data)
        print(lte_id_data.shape)
        # adaboost 训练
        adaboost_main(lte_id_data,BS_lng,BS_lat,limli_scale,lte_id)
        print(lte_id)
    np.savez('data/BS.npz', BS_lng_all, BS_lat_all)

def pred():  # 预测数据
    test_data = pd.read_csv('没有经纬度的华为数据.csv', encoding='ANSI')
    # 准备格子
    grid = pd.read_csv('栅格的中心经纬度.csv',encoding='ANSI')
    grid_x = np.array(list(set(np.array(grid)[:,0])))
    grid_x.sort()
    grid_y = np.array(list(set(np.array(grid)[:,1])))
    grid_y.sort()

    ##
    grid_x = np.linspace(103.6865+0.00055,103.7146-0.00055,25)
    grid_y = np.linspace(36.067044+0.00045,36.08712-0.00045,22)


    limli_scale_data = pd.read_csv('计算覆盖距离.csv', encoding='ANSI')
    train_lng,train_lat,pred_ada_lng,pred_ada_lat,pred_knn_lng,pred_knn_lat,pred_ada_knn_lng,pred_ada_knn_lat\
        = [],[],[],[],[],[],[],[]
    time_start = 0
    for index, row in limli_scale_data.iterrows():
        ## 准备训练数据
        lte_id = row['lte_id']
        print(lte_id)
        data = np.load('data/{}_应用集.npz'.format(lte_id))
        lte_id_data = data['arr_0']
        # 准备预测数据
        id_test_data = test_data[test_data['lte_id'] == lte_id]
        predict_x = id_test_data[
            ['ltescrsrp', 'ltescrsrq', 'ltescphr', 'ltencrsrp_1', 'ltencrsrq_1', 'ltencpci_1', 'ltencrsrp_2',
             'ltencrsrq_2', 'ltencpci_2']]
        predict_x = np.array(predict_x)  ## 准备需要预测的数据

        ###
        predict_x = predict_x.T
        train_data = np.array(lte_id_data).T  # (25721,19)
        train_y = train_data[9:, ]
        train_y_lng = train_y[0, :]
        train_y_lat = train_y[1, :]
        train_lng.extend(train_y_lng)
        train_lat.extend(train_y_lat)
        ###
        ada_lng, ada_lat, knn_lng, knn_lat, ada_knn_lng, ada_knn_lat,time_scale = data_op_main(predict_x,lte_id_data,lte_id)
        time_start+=time_scale
        pred_ada_lng.extend(ada_lng)
        pred_ada_lat.extend(ada_lat)
        pred_knn_lng.extend(knn_lng)
        pred_knn_lat.extend(knn_lat)
        pred_ada_knn_lng.extend(ada_knn_lng)
        pred_ada_knn_lat.extend(ada_knn_lat)

    ####格式化格子
    for i in range(len(pred_ada_lng)):
        form_lng = grid_x[np.argmin(np.abs(pred_ada_lng[i] - grid_x))]
        form_lat = grid_y[np.argmin(np.abs(pred_ada_lat[i] - grid_y))]
        pred_ada_lng[i] = form_lng
        pred_ada_lat[i] = form_lat

        form_lng = grid_x[np.argmin(np.abs(pred_knn_lng[i] - grid_x))]
        form_lat = grid_y[np.argmin(np.abs(pred_knn_lat[i] - grid_y))]
        pred_knn_lng[i] = form_lng
        pred_knn_lat[i] = form_lat

        form_lng = grid_x[np.argmin(np.abs(pred_ada_knn_lng[i] - grid_x))]
        form_lat = grid_y[np.argmin(np.abs(pred_ada_knn_lat[i] - grid_y))]
        pred_ada_knn_lng[i] = form_lng
        pred_ada_knn_lat[i] = form_lat


    xx, yy = np.meshgrid(grid_x, grid_y)
    xxx, yyy = xx.flatten(), yy.flatten()
    trp_train = {}
    for i,j in zip(xxx,yyy):
        trp_train[(i,j)] = 0
    for i in range(len(train_lng)):
        form_lng = grid_x[np.argmin(np.abs(train_lng[i] - grid_x))]
        form_lat = grid_y[np.argmin(np.abs(train_lat[i] - grid_y))]
        train_lng[i] = form_lng
        train_lat[i] = form_lat
        trp_train[(form_lng, form_lat)] = 1
    print('===',len(train_lng))

    see_ada,not_see_ada,see_ada_knn,not_see_ada_knn = 0,0,0,0
    for i in range(len(pred_ada_lng)):
        if trp_train[(pred_ada_lng[i],pred_ada_lat[i])]==1 :
            see_ada+=1
        else:not_see_ada+=1
        if trp_train[(pred_ada_knn_lng[i],pred_ada_knn_lat[i])]==1:
            see_ada_knn+=1
        else:not_see_ada_knn+=1
    print('see_ada',see_ada)
    print('see_ada_knn',see_ada_knn)
    print('not_see_ada',not_see_ada)
    print('not_see_ada_knn',not_see_ada_knn)
    print('time','time_one',time_start,time_start/len(pred_ada_knn_lng))
    print(len(pred_ada_knn_lng))
    ###
    print('===========================')
    data = np.load('data/BS.npz')
    BS_lng = data['arr_0']
    BS_lat = data['arr_1']

    plt.xlim((103.685, 103.710))  # 设置坐标轴范围
    plt.ylim((36.0675, 36.0875))
    plt.scatter(BS_lng, BS_lat, c='r')
    plt.scatter(pred_ada_lng, pred_ada_lat, c='b')
    plt.scatter(train_lng,train_lat,c='g')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend(['基站位置','adaboost算法对应用集预测结果','训练数据点'],loc='lower left')
    plt.show()

    # plt.xlim((103.685, 103.710))  # 设置坐标轴范围
    # plt.ylim((36.0675, 36.0875))
    # plt.scatter(BS_lng, BS_lat, c='r')
    # plt.scatter(pred_knn_lng, pred_knn_lat, c='b')
    # plt.scatter(train_lng,train_lat,c='g')
    # plt.xlabel('经度')
    # plt.ylabel('纬度')
    # plt.legend(['基站位置','KNN算法','带经纬度点'],loc='lower left')
    # plt.show()

    plt.xlim((103.685, 103.710))  # 设置坐标轴范围
    plt.ylim((36.0675, 36.0875))
    plt.scatter(BS_lng, BS_lat, c='r')
    plt.scatter(pred_ada_knn_lng, pred_ada_knn_lat, c='b')
    plt.scatter(train_lng,train_lat,c='g')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend(['基站位置','加权平均算法对应用集预测结果','训练数据点'],loc='lower left')
    plt.show()

# train()
pred()

