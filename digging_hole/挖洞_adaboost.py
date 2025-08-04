import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt
def adaboost_main(data,BS_lng,BS_lat,limli_scale,lte_id,lng_1,lat_1,lng_2,lat_2,lng_3,lat_3):
    def rad(d):
        return d * np.pi / 180.0
    data = np.array(data)
    data = data.T
    x = data[0:9,:]
    y = data[9:,]
    y_lng = y[0,:]
    y_lat = y[1,:]
    lng1 = BS_lng
    lng2 = y_lng
    lat1 = BS_lat
    lat2 = y_lat
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * np.arcsin(np.sqrt(pow(np.sin(a/2), 2) + np.cos(radLat1) * np.cos(radLat2) * pow(np.sin(b/2), 2)))
    s = s * 6378.137 * 1000
    num = []
    for j in range(len(s)):
        if s[j] > limli_scale:
            num.append(j)
    print('500米范围数据占：',1-len(num)/len(s))
    x = np.delete(x,num,axis=1)
    y = np.delete(y,num,axis=1)
    x = x.T
    scaler = preprocessing.MinMaxScaler()
    x = scaler.fit_transform(x)
    y = y.T
    ###
    print(x.shape,y.shape)
    y_lng = y[:,0]
    y_lat = y[:,1]
    dig_1,dig_2,dig_3 = 0,0,0
    for i in range(len(y_lat)):
        if (lat_1<y_lat[i]<lat_1+0.00045) and (lng_1<y_lng[i]<lng_1+0.00055):
           dig_1+=1
        if (lat_2<y_lat[i]<lat_2+0.00045) and (lng_2<y_lng[i]<lng_2+0.00055):dig_2+=1
        if (lat_3 < y_lat[i] < lat_3 + 0.00045) and (lng_3 < y_lng[i] < lng_3 + 0.00055):dig_3+=1
    print('dig_1:',dig_1,'dig_2:',dig_2,'dig_3:',dig_3)

    num_test = []
    num_train = []
    for i in range(len(y_lat)):
        if (lat_1<y_lat[i]<lat_1+0.00045) and (lng_1<y_lng[i]<lng_1+0.00055) or (lat_2<y_lat[i]<lat_2+0.00045) and (lng_2<y_lng[i]<lng_2+0.00055)\
                or (lat_3 < y_lat[i] < lat_3 + 0.00045) and (lng_3 < y_lng[i] < lng_3 + 0.00055) :
           num_test.append(i)
        else:num_train.append(i)

    train_x = x[num_train]
    train_y = y[num_train]
    test_x = x[num_test]
    test_y = y[num_test]
    print('train：',train_y.shape,'test:',test_y.shape)
    plt.scatter(train_y[:,0],train_y[:,1],s=0.5,c='r')
    plt.scatter(test_y[:,0],test_y[:,1],s=0.5,c='b')

    ###
    bdt_lng = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(criterion='mse',splitter='random',max_depth=31,
                                                                 min_samples_split=2,min_samples_leaf=1),
                            n_estimators=98,learning_rate=0.01,loss='square')
    bdt_lng.fit(train_x, train_y[:,0])
    train_y_pred = bdt_lng.predict(train_x)
    test_y_pred = bdt_lng.predict(test_x)
    # print('训练经度误差:',np.mean(np.abs(train_y_pred-train_y[:,0])))
    # print('测试经度误差:',np.mean(np.abs(test_y_pred-test_y[:,0])))
    lng1 = test_y_pred
    lng2 = test_y[:,0]
    #--------------------多加一个特征-------------------------#
    train_y_pred = train_y_pred.reshape(train_x.shape[0],1)
    train_x_lat = np.hstack((train_x,train_y_pred))
    test_y_pred = test_y_pred.reshape(test_x.shape[0],1)
    test_x_lat = np.hstack((test_x,test_y_pred))
    #-----------------------------------------------------------#
    bdt_lat = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(criterion='mse',splitter='random',max_depth=31),
                            #, min_samples_split=20, min_samples_leaf=5),
                            n_estimators=98,learning_rate=0.01,loss='square')
    bdt_lat.fit(train_x_lat, train_y[:,1])
    train_y_pred = bdt_lat.predict(train_x_lat)
    # print('训练纬度误差:',np.mean(np.abs(train_y_pred-train_y[:,1])))
    test_y_pred = bdt_lat.predict(test_x_lat)
    # print('测试纬度误差:',np.mean(np.abs(test_y_pred-test_y[:,1])))
    lat1 = test_y_pred
    lat2 = test_y[:,1]
    def rad(d):
        return d * np.pi / 180.0
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * np.arcsin(np.sqrt(pow(np.sin(a/2), 2) + np.cos(radLat1) * np.cos(radLat2) * pow(np.sin(b/2), 2)))
    s = s * 6378.137 * 1000
    # print(lng1.shape,lat1.shape)
    plt.scatter(lng1, lat1, s=0.5, c='g')
    plt.show()
    i = 0
    for j in range(len(lng1)):
        if ((lat_1<lat1[i]<lat_1+0.00045) and (lng_1<lng1[i]<lng_1+0.00055)) or ((lat_2<lat1[i]<lat_2+0.00045) and (lng_2<lng1[i]<lng_2+0.00055))\
                or ((lat_3 < lat1[i] < lat_3 + 0.00045) and (lng_3 < lng1[i] < lng_3 + 0.00055)) :
           i+=1
    print('adaboost:','train:',train_y.shape,'test:',test_y.shape,'digit:',i)
    ###  保存模型


def train_adaboost(file,lng_1,lat_1,lng_2,lat_2,lng_3,lat_3): ## 训练模型
    limli_scale_data=pd.read_csv('计算覆盖距离.csv',encoding='ANSI')
    data_with_GPS = pd.read_excel(file)

    lte_id = file.split('.')[0]
    limli_scale = 500
    BS_lng = data_with_GPS[data_with_GPS['lte_id'] == lte_id].经度.iloc[0]
    BS_lat = data_with_GPS[data_with_GPS['lte_id'] == lte_id].维度.iloc[0]
    lte_id_data = data_with_GPS[data_with_GPS['lte_id'] == lte_id][['ltescrsrp','ltescrsrq','ltescphr','ltencrsrp_1','ltencrsrq_1','ltencpci_1','ltencrsrp_2','ltencrsrq_2','ltencpci_2','longitude','latitude']]

    # adaboost 训练
    adaboost_main(lte_id_data,BS_lng,BS_lat,limli_scale,lte_id,lng_1,lat_1,lng_2,lat_2,lng_3,lat_3)
    # print(lte_id)
# train_adaboost('183053-1.xlsx',103.704,36.074)

