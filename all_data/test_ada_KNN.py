import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from All_data.adaboost import adaboost_main
from All_data.data_op_ada_KNN import data_op_main

# from All_data.adaboost_从家 import adaboost_main
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']


### 用于数据集的画图。数据的划分仍然是百分之20.
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
    limli_scale_data=pd.read_csv('计算覆盖距离.csv',encoding='ANSI')
    data_with_GPS = pd.read_excel('华为已处理数据.xlsx')
    BS_lng_all,BS_lat_all=[],[]
    for index,row in limli_scale_data.iterrows():
        lte_id = row['lte_id']
        limli_scale = row['覆盖范围']
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
        train_x, test_x, train_y, test_y = train_test_split(lte_id_data[:,0:9], lte_id_data[:,9:], test_size=0.2)
        print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
        lte_id_data = np.concatenate((train_x,train_y),axis=1)
        test_data  = np.concatenate((test_x,test_y),axis=1)
        np.savez('data/{}.npz'.format(lte_id),lte_id_data,test_data)
        print(lte_id_data.shape)
        # adaboost 训练
        adaboost_main(lte_id_data,BS_lng,BS_lat,limli_scale,lte_id)
        print(lte_id)
    np.savez('data/BS.npz', BS_lng_all, BS_lat_all)

def pred():  # 预测数据
    # data = pd.read_csv('没有经纬度的华为数据.csv', encoding='ANSI')
    # 准备格子
    grid = pd.read_csv('栅格的中心经纬度.csv',encoding='ANSI')
    grid_x = np.array(list(set(np.array(grid)[:,0])))
    grid_x.sort()
    grid_y = np.array(list(set(np.array(grid)[:,1])))
    grid_y.sort()

    grid_x = np.linspace(103.6865+0.00055,103.7146-0.00055,25)
    grid_y = np.linspace(36.067044+0.00045,36.08712-0.00045,22)


    limli_scale_data = pd.read_csv('计算覆盖距离.csv', encoding='ANSI')
    # data_with_GPS = pd.read_excel('华为已处理数据.xlsx')
    s_ada_ls,s_ada_knn_ls,s_knn_ls = [],[],[]
    s_ada_10_ls, s_ada_30_ls, s_ada_50_ls, s_knn_10_ls, s_knn_30_ls, s_knn_50_ls, s_ada_knn_10_ls, s_ada_knn_30_ls, s_ada_knn_50_ls=[],[],[],[],[],[],[],[],[]
    train_lng,train_lat,test_lng,test_lat,pred_ada_lng,pred_ada_lat,pred_knn_lng,pred_knn_lat,pred_ada_knn_lng,pred_ada_knn_lat\
        = [],[],[],[],[],[],[],[],[],[]
    for index, row in limli_scale_data.iterrows():
        ## 准备训练数据
        lte_id = row['lte_id']
        data = np.load('data/{}.npz'.format(lte_id))
        lte_id_data = data['arr_0']
        # 准备预测数据
        predict_x = data['arr_1']
        ###
        test_x = predict_x.T
        test_y = test_x[9:, ]
        train_data = np.array(lte_id_data).T  # (25721,19)
        train_y = train_data[9:, ]
        train_y_lng = train_y[0, :]
        train_y_lat = train_y[1, :]
        train_lng.extend(train_y_lng)
        train_lat.extend(train_y_lat)
        test_lng.extend(test_y[0, :])
        test_lat.extend(test_y[1,:])
        ###
        ada_lng, ada_lat, knn_lng, knn_lat, ada_knn_lng, ada_knn_lat, s_ada, s_knn, s_ada_knn,s_rate = data_op_main(predict_x,lte_id_data,lte_id,grid_x,grid_y)
        pred_ada_lng.extend(ada_lng)
        pred_ada_lat.extend(ada_lat)
        pred_knn_lng.extend(knn_lng)
        pred_knn_lat.extend(knn_lat)
        pred_ada_knn_lng.extend(ada_knn_lng)
        pred_ada_knn_lat.extend(ada_knn_lat)
        s_ada_ls.append(s_ada)
        s_ada_knn_ls.append(s_ada_knn)
        s_knn_ls.append(s_knn)
        s_ada_10, s_ada_30, s_ada_50, s_knn_10, s_knn_30, s_knn_50, s_ada_knn_10, s_ada_knn_30, s_ada_knn_50 = s_rate
        s_ada_10_ls.append(s_ada_10)
        s_ada_30_ls.append(s_ada_30)
        s_ada_50_ls.append(s_ada_50)
        s_knn_10_ls.append(s_knn_10)
        s_knn_30_ls.append(s_knn_30)
        s_knn_50_ls.append(s_knn_50)
        s_ada_knn_10_ls.append(s_ada_knn_10)
        s_ada_knn_30_ls.append(s_ada_knn_30)
        s_ada_knn_50_ls.append(s_ada_knn_50)
    print('===========================')
    print('ada',np.mean(s_ada_ls))
    print('ada_knn',np.mean(s_ada_knn_ls))
    print('knn',np.mean(s_knn_ls))
    print('ada_10',np.mean(s_ada_10_ls))
    print('ada_30', np.mean(s_ada_30_ls))
    print('ada_50', np.mean(s_ada_50_ls))
    print('knn_10', np.mean(s_knn_10_ls))
    print('knn_30', np.mean(s_knn_30_ls))
    print('knn_50', np.mean(s_knn_50_ls))
    print('ada_knn_10', np.mean(s_ada_knn_10_ls))
    print('ada_knn_30', np.mean(s_ada_knn_30_ls))
    print('ada_knn_50', np.mean(s_ada_knn_50_ls))

    data = np.load('data/BS.npz')
    BS_lng = data['arr_0']
    BS_lat = data['arr_1']


    ####格式化格子
    # for i in range(len(test_lng)):
    #     form_lng = grid_x[np.argmin(np.abs(test_lng[i]-grid_x))]
    #     form_lat = grid_y[np.argmin(np.abs(test_lat[i] - grid_y))]
    #     test_lng[i] = form_lng
    #     test_lat[i] = form_lat
    #
    #     form_lng = grid_x[np.argmin(np.abs(pred_ada_lng[i]-grid_x))]
    #     form_lat = grid_y[np.argmin(np.abs(pred_ada_lat[i] - grid_y))]
    #     pred_ada_lng[i] = form_lng
    #     pred_ada_lat[i] = form_lat
    #
    #     form_lng = grid_x[np.argmin(np.abs(pred_knn_lng[i]-grid_x))]
    #     form_lat = grid_y[np.argmin(np.abs(pred_knn_lat[i] - grid_y))]
    #     pred_knn_lng[i] = form_lng
    #     pred_knn_lat[i] = form_lat
    #
    #     form_lng = grid_x[np.argmin(np.abs(pred_ada_knn_lng[i]-grid_x))]
    #     form_lat = grid_y[np.argmin(np.abs(pred_ada_knn_lat[i] - grid_y))]
    #     pred_ada_knn_lng[i] = form_lng
    #     pred_ada_knn_lat[i] = form_lat
    #检验是否同分布
    import scipy.stats
    def JS_divergence(p, q):
        M = (p + q) / 2
        return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)
    pred_knn = np.concatenate((np.array(pred_knn_lng).reshape(len(pred_knn_lng),1),np.array(pred_knn_lat).reshape(len(pred_knn_lat),1)),axis=1)
    pred_ada = np.concatenate(
        (np.array(pred_ada_lng).reshape(len(pred_ada_lng), 1), np.array(pred_ada_lat).reshape(len(pred_ada_lat), 1)),
        axis=1)
    pred_ada_knn = np.concatenate(
        (np.array(pred_ada_knn_lng).reshape(len(pred_ada_knn_lng), 1), np.array(pred_ada_knn_lat).reshape(len(pred_ada_knn_lat), 1)),
        axis=1)
    pred_test = np.concatenate(
        (np.array(test_lng).reshape(len(test_lng), 1), np.array(test_lat).reshape(len(test_lat), 1)),
        axis=1)
    pred_train = np.concatenate(
        (np.array(train_lng).reshape(len(train_lng), 1), np.array(train_lat).reshape(len(train_lat), 1)),
        axis=1)
    print(pred_knn.shape)
    print('JS',JS_divergence(pred_test,pred_ada))
    print('JS', JS_divergence(pred_test, pred_knn))
    print('JS', JS_divergence(pred_test, pred_ada_knn))
    # print('JS', JS_divergence(pred_train, pred_test))

    train_lng.extend(test_lng)
    train_lat.extend(test_lat)
    grid_x = np.linspace(103.6865+0.00055,103.7146-0.00055,25)
    grid_y = np.linspace(36.067044+0.00045,36.08712-0.00045,22)
    print('=====',len(train_lng))
    for i in range(len(train_lng)):
        form_lng = grid_x[np.argmin(np.abs(train_lng[i]-grid_x))]
        form_lat = grid_y[np.argmin(np.abs(train_lat[i] - grid_y))]
        train_lng[i] = form_lng
        train_lat[i] = form_lat
    plt.xlim((103.685, 103.710))  # 设置坐标轴范围
    plt.ylim((36.0675, 36.0875))
    plt.scatter(train_lng,train_lat,c='b',s=0.5)
    plt.scatter(BS_lng, BS_lat, c='r',s=5)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend(['训练集','基站'],loc='lower left')
    plt.show()

    plt.figure(figsize=(10,8))
    plt.subplot(221)
    plt.xlim((103.685, 103.710))  # 设置坐标轴范围
    plt.ylim((36.0675, 36.0875))
    plt.scatter(test_lng,test_lat,c='g',s=0.5)
    plt.scatter(BS_lng, BS_lat, c='r',s=5)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend(['测试集的真实经纬度点','基站点'],loc='lower left')
    # plt.show()

    '''plt.subplot(222)
    plt.xlim((103.685, 103.710))  # 设置坐标轴范围
    plt.ylim((36.0675, 36.0875))
    # plt.scatter(test_lng,test_lat,c='b',s=0.5)
    plt.scatter(pred_ada_lng,pred_ada_lat,c='b',s=0.5)
    plt.scatter(BS_lng, BS_lat, c='r',s=5)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend(['AdaBoost算法的预测点','基站点'],loc='lower left')'''
    # plt.show()

    '''plt.subplot(223)
    plt.xlim((103.685, 103.710))  # 设置坐标轴范围
    plt.ylim((36.0675, 36.0875))
    # plt.scatter(test_lng,test_lat,c='b',s=0.5)
    plt.scatter(pred_knn_lng,pred_knn_lat,c='b',s=0.5)
    plt.scatter(BS_lng, BS_lat, c='r',s=5)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend(['KNN算法的预测点','基站点'],loc='lower left')'''
    # plt.show()

    plt.subplot(222)
    plt.xlim((103.685, 103.710))  # 设置坐标轴范围
    plt.ylim((36.0675, 36.0875))
    # plt.scatter(test_lng,test_lat,c='b',s=0.5)
    plt.scatter(pred_ada_knn_lng,pred_ada_knn_lat,c='b',s=0.5)
    plt.scatter(BS_lng, BS_lat, c='r',s=5)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend(['加权平均算法的预测点','基站点'],loc='lower left')
    plt.show()
# train()
pred()
