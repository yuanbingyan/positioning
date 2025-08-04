import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn import preprocessing
import os
def adaboost_main(data,BS_lng,BS_lat,limli_scale,lte_id):
    def rad(d):
        return d * np.pi / 180.0
    # data = pd.read_excel('180757-1.xlsx')
    data = np.array(data)
    data = data.T
    x = data[0:9,:]
    y = data[9:,]
    # BS_lng = 103.700985
    # BS_lat = 36.075353
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
    # print('500米范围数据占：',1-len(num)/len(s))
    x = np.delete(x,num,axis=1)
    y = np.delete(y,num,axis=1)
    x = x.T
    scaler = preprocessing.MinMaxScaler()
    x = scaler.fit_transform(x)
    y = y.T
    train_x,test_x, train_y, test_y =train_test_split(x,y,test_size=0.1, random_state=0)
    bdt_lng = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(criterion='friedman_mse',splitter='random',max_depth=31,
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
    bdt_lat = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(criterion='friedman_mse',splitter='random',max_depth=31),
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
    ###  保存模型
    if not os.path.exists('adaboost_model/'):
        os.makedirs('adaboost_model/')
    joblib.dump((scaler,bdt_lng, bdt_lat),'adaboost_model/'+lte_id+'.pkl')


###测试模型精确度
# data = pd.read_excel('180757-1.xlsx')
# data = np.array(data) #(25721,19)
# data = data.T# (19, 25721)
# x = data[0:9,:] # (17, 25721)
# y = data[9:,]
# y_lng = y[0,:]
# y_lat = y[1,:]
# x = x.T
# x = scaler.transform(x)
# lng = bdt_lng.predict(x)
# train_y_pred = lng.reshape(x.shape[0],1)
# train_x_lat = np.hstack((x,train_y_pred))
# lat = bdt_lat.predict(train_x_lat)
# print(np.mean(np.abs(lng-y_lng)))
# print(np.mean(np.abs(lat-y_lat)))




