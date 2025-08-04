import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from All_data.adaboost import adaboost_main
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
from All_data.挖洞.挖洞_使用全部小区_data_op_AdaBoost import data_op_main
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
def train(lng_1,lat_1,lng_2,lat_2,lng_3,lat_3,lng_4,lat_4,lng_5,lat_5): ## 训练模型
    limli_scale_data = pd.read_csv('计算覆盖距离.csv',encoding='ANSI')
    data_with_GPS = pd.read_excel('华为已处理数据.xlsx')
    BS_lng_all,BS_lat_all=[],[]
    for index,row in limli_scale_data.iterrows():
        lte_id = row['lte_id']
        limli_scale = row['覆盖范围']
        limli_scale=500
        # if limli_scale<=500:
        #     limli_scale = 500
        BS_lng = data_with_GPS[data_with_GPS['lte_id'] == lte_id].经度.iloc[0]
        BS_lat = data_with_GPS[data_with_GPS['lte_id'] == lte_id].维度.iloc[0]
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
        ###
        num_test = []
        num_train = []
        for i in range(len(train_y[:,0])):
            if (lat_1 < train_y[i,1] < lat_1 + 0.00045) and (lng_1 < train_y[i,0] < lng_1 + 0.00055) or (
                    lat_2 < train_y[i,1] < lat_2 + 0.00045) and (lng_2 < train_y[i,0] < lng_2 + 0.00055) \
                    or (lat_3 < train_y[i,1] < lat_3 + 0.00045) and (lng_3 < train_y[i,0] < lng_3 + 0.00055)or\
                (lat_4 < train_y[i,1] < lat_4 + 0.00045) and (lng_4 < train_y[i,0] < lng_4 + 0.00055) or\
                (lat_5 < train_y[i,1] < lat_5 + 0.00045) and (lng_5 < train_y[i,0] < lng_5 + 0.00055):
                num_test.append(i)
            else:
                num_train.append(i)
        test_x = train_x[num_test]
        train_x = train_x[num_train]
        test_y = train_y[num_test]
        train_y = train_y[num_train]

        print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
        lte_id_data = np.concatenate((train_x,train_y),axis=1)
        test_data = test_x
        np.savez('data/{}_应用集.npz'.format(lte_id),lte_id_data,test_data)
        print(lte_id_data.shape)
        # adaboost 训练
        adaboost_main(lte_id_data,BS_lng,BS_lat,limli_scale,lte_id)
        print(lte_id)
    np.savez('data/BS.npz', BS_lng_all, BS_lat_all)

def pred():  # 预测数据
    # 准备格子
    grid = pd.read_csv('栅格的中心经纬度.csv',encoding='ANSI')
    grid_x = np.array(list(set(np.array(grid)[:,0])))
    grid_x.sort()
    grid_y = np.array(list(set(np.array(grid)[:,1])))
    grid_y.sort()

    ##
    grid_x = np.linspace(103.6865+0.00055,103.7146-0.00055,50)
    grid_y = np.linspace(36.067044+0.00045,36.08712-0.00045,44)


    limli_scale_data = pd.read_csv('计算覆盖距离.csv', encoding='ANSI')
    train_lng,train_lat,pred_ada_lng,pred_ada_lat,pred_knn_lng,pred_knn_lat,pred_ada_knn_lng,pred_ada_knn_lat\
        = [],[],[],[],[],[],[],[]
    train_num,test_num=0,0
    for index, row in limli_scale_data.iterrows():
        ## 准备训练数据
        lte_id = row['lte_id']
        print(lte_id)
        data = np.load('data/{}_应用集.npz'.format(lte_id))
        lte_id_data = data['arr_0']
        # 准备预测数据
        test_data = data['arr_1']
        train_num+=len(lte_id_data)
        test_num+=len(test_data)
        id_test_data = test_data
        predict_x = id_test_data
        predict_x = np.array(predict_x)  ## 准备需要预测的数据
        if len(predict_x)==0:continue
        print(len(predict_x))
        ###
        predict_x = predict_x.T
        train_data = np.array(lte_id_data).T  # (25721,19)
        train_y = train_data[9:, ]
        train_y_lng = train_y[0, :]
        train_y_lat = train_y[1, :]
        train_lng.extend(train_y_lng)
        train_lat.extend(train_y_lat)
        ###
        ada_lng, ada_lat, knn_lng, knn_lat, ada_knn_lng, ada_knn_lat = data_op_main(predict_x,lte_id_data,lte_id)

        pred_ada_lng.extend(ada_lng)
        pred_ada_lat.extend(ada_lat)
        pred_knn_lng.extend(knn_lng)
        pred_knn_lat.extend(knn_lat)
        pred_ada_knn_lng.extend(ada_knn_lng)
        pred_ada_knn_lat.extend(ada_knn_lat)


##
    ada_dig = 0
    ada_not_dig = 0
    for i in range(len(pred_ada_lng)):
        if (lat_1 < pred_ada_lat[i] < lat_1 + 0.00045) and (lng_1 < pred_ada_lng[i] < lng_1 + 0.00055) or (
                lat_2 < pred_ada_lat[i] < lat_2 + 0.00045) and (lng_2 < pred_ada_lng[i] < lng_2 + 0.00055) \
                or (lat_3 < pred_ada_lat[i] < lat_3 + 0.00045) and (lng_3 < pred_ada_lng[i] < lng_3 + 0.00055)\
                or (lat_4 < pred_ada_lat[i] < lat_4 + 0.00045) and (lng_4 < pred_ada_lng[i] < lng_4 + 0.00055)\
                or (lat_5 < pred_ada_lat[i] < lat_5 + 0.00045) and (lng_5 < pred_ada_lng[i] < lng_5 + 0.00055):
            ada_dig+=1
        else:
            ada_not_dig+=1
    ada_knn_dig = 0
    ada_knn_not_dig = 0
    for i in range(len(pred_ada_lng)):
        if (lat_1 < pred_ada_knn_lat[i] < lat_1 + 0.00045) and (lng_1 < pred_ada_knn_lng[i] < lng_1 + 0.00055) or (
                lat_2 < pred_ada_knn_lat[i] < lat_2 + 0.00045) and (lng_2 < pred_ada_knn_lng[i] < lng_2 + 0.00055) \
                or (lat_3 < pred_ada_knn_lat[i] < lat_3 + 0.00045) and (lng_3 < pred_ada_knn_lng[i] < lng_3 + 0.00055)\
                or (lat_4 < pred_ada_knn_lat[i] < lat_4 + 0.00045) and (lng_4 < pred_ada_knn_lng[i] < lng_4 + 0.00055)\
                or (lat_5 < pred_ada_knn_lat[i] < lat_5 + 0.00045) and (lng_5 < pred_ada_knn_lng[i] < lng_5 + 0.00055):
            ada_knn_dig+=1
        else:
            ada_knn_not_dig+=1
    knn_dig = 0
    knn_not_dig = 0
    for i in range(len(pred_knn_lng)):
        if (lat_1 < pred_knn_lat[i] < lat_1 + 0.00045) and (lng_1 < pred_knn_lng[i] < lng_1 + 0.00055) or (
                lat_2 < pred_knn_lat[i] < lat_2 + 0.00045) and (lng_2 < pred_knn_lng[i] < lng_2 + 0.00055) \
                or (lat_3 < pred_knn_lat[i] < lat_3 + 0.00045) and (lng_3 < pred_knn_lng[i] < lng_3 + 0.00055) \
                or (lat_4 < pred_knn_lat[i] < lat_4 + 0.00045) and (lng_4 < pred_knn_lng[i] < lng_4 + 0.00055)\
                or (lat_5 < pred_knn_lat[i] < lat_5 + 0.00045) and (lng_5 < pred_knn_lng[i] < lng_5 + 0.00055):
            knn_dig+=1
        else:
            knn_not_dig+=1
    print('train_num',train_num)
    print('test_num',test_num)
    print('ada_dig',ada_dig)
    print('ada_not_dig',ada_not_dig)
    print('knn_dig',knn_dig)
    print('knn_not_dig',knn_not_dig)
    print('ada_knn_dig',ada_knn_dig)
    print('ada_knn_not_dig',ada_knn_not_dig)
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


    ###
    print('===========================')
    data = np.load('data/BS.npz')
    BS_lng = data['arr_0']
    BS_lat = data['arr_1']

    # plt.xlim((103.685, 103.710))  # 设置坐标轴范围
    # plt.ylim((36.0675, 36.0875))
    plt.scatter(BS_lng, BS_lat, c='r')
    plt.scatter(pred_ada_lng, pred_ada_lat, c='b',s=3)
    plt.scatter(train_lng,train_lat,c='g',s=3)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend(['基站','adaboost算法','带经纬度点'],loc='lower left')
    plt.show()

    # plt.xlim((103.685, 103.710))  # 设置坐标轴范围
    # plt.ylim((36.0675, 36.0875))
    plt.scatter(BS_lng, BS_lat, c='r')
    plt.scatter(pred_knn_lng, pred_knn_lat, c='b',s=3)
    plt.scatter(train_lng,train_lat,c='g',s=3)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend(['基站','KNN算法','带经纬度点'],loc='lower left')
    plt.show()

    # plt.xlim((103.685, 103.710))  # 设置坐标轴范围
    # plt.ylim((36.0675, 36.0875))
    plt.scatter(BS_lng, BS_lat, c='r')
    plt.scatter(pred_ada_knn_lng, pred_ada_knn_lat, c='b',s=3)
    plt.scatter(train_lng,train_lat,c='g',s=3)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend(['基站','AdaBoost与KNN加权平均','带经纬度点'],loc='lower left')
    plt.show()

lng = np.random.uniform(103.695,103.709,5)
lat = np.random.uniform(36.0725,36.08,5)


lng_1 = lng[0]
lat_1 = lat[0]
lng_2 = lng[1]
lat_2 = lat[1]
lng_3 = lng[2]
lat_3 = lat[2]
lng_4 = lng[3]
lat_4 = lat[3]
lng_5 = lng[4]
lat_5 = lat[4]
train(lng_1,lat_1,lng_2,lat_2,lng_3,lat_3,lng_4,lat_4,lng_5,lat_5)
pred()

