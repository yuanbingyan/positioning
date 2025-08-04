import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
### 给出经纬度计算距离
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
k = 1
knn = KNeighborsRegressor(k)
###

def data_op_main(predict_x,train_data,lte_id,grid_x,grid_y):
    predict_x = predict_x.T
    predict_y = predict_x[9:,]
    predict_x = predict_x[0:9,].T  ## 准备需要预测的数据
    train_data = np.array(train_data).T #(25721,19)
    train_x = train_data[0:9, ].T
    train_y = train_data[9:,]
    train_y_lng = train_y[0,:]
    train_y_lat = train_y[1,:]
    ### knn
    knn.fit(train_x, train_y_lng)
    knn_lng = knn.predict(predict_x)
    knn.fit(train_x, train_y_lat)
    knn_lat = knn.predict(predict_x)
    s_knn = distance(knn_lng, predict_y[0, :], knn_lat, predict_y[1, :])

    ###
    xx, yy = np.meshgrid(grid_x, grid_y)
    xxx, yyy = xx.flatten(), yy.flatten()
    pos = {}
    for i,j in zip(xxx,yyy):
        pos[(i,j)] = 0
    ##### 把训练数据对应到网格
    for i in range(len(train_y_lat)):
        form_lng = grid_x[np.argmin(np.abs(train_y_lng[i]-grid_x))]
        form_lat = grid_y[np.argmin(np.abs(train_y_lat[i] - grid_y))]
        train_y_lng[i] = form_lng
        train_y_lat[i] = form_lat
        pos[(form_lng,form_lat)] = 1

    ###
    (scaler_adaboost, bdt_lng, bdt_lat) = joblib.load('adaboost_model/' + lte_id + '.pkl')
    adaboost_x = predict_x
    adaboost_x = scaler_adaboost.transform(adaboost_x)
    ada_lng = bdt_lng.predict(adaboost_x)
    train_x_addFeature = ada_lng.reshape(adaboost_x.shape[0], 1)
    train_x_addFeature = np.hstack((adaboost_x, train_x_addFeature))
    ada_lat = bdt_lat.predict(train_x_addFeature)
    s_ada = distance(ada_lng, predict_y[0, :], ada_lat, predict_y[1, :])

    w = 0.8
    s_ada_knn = distance(w * ada_lng+(1-w)*knn_lng, predict_y[0, :], w*ada_lat +(1-w)*knn_lat, predict_y[1, :])
    ada_knn_lng = w * ada_lng + (1-w) * knn_lng
    ada_knn_lat = w * ada_lat + (1-w) * knn_lat
 ###



    s_ada_10,s_ada_30,s_ada_50 = np.sum(s_ada<17)/len(s_ada),np.sum(s_ada<30)/len(s_ada),np.sum(s_ada<50)/len(s_ada)
    s_knn_10, s_knn_30, s_knn_50 = np.sum(s_knn < 17) / len(s_knn), np.sum(s_knn < 30) / len(s_knn), np.sum(
        s_knn < 50) / len(s_knn)
    s_ada_knn_10, s_ada_knn_30, s_ada_knn_50 = np.sum(s_ada_knn < 17) / len(s_ada_knn), np.sum(s_ada_knn < 30) / len(s_ada_knn), np.sum(
        s_ada_knn < 50) / len(s_ada_knn)
    print('lte_id:',lte_id,'s_ada_10',round(s_ada_10,3),'s_ada_30',round(s_ada_30,3),'s_ada_50',round(s_ada_50,3),)
    print('lte_id:',lte_id,'s_knn_10',round(s_knn_10,3),'s_knn_30',round(s_knn_30,3),'s_knn_50',round(s_knn_50,3),)
    print('lte_id:',lte_id,'s_ada_knn_10',round(s_ada_knn_10,3),'s_ada_knn_30',round(s_ada_knn_30,3),'s_ada_knn_50',round(s_ada_knn_50,2),)
    s_rate = [s_ada_10,s_ada_30,s_ada_50,s_knn_10,s_knn_30,s_knn_50,s_ada_knn_10,s_ada_knn_30,s_ada_knn_50]

    # print('s_train_knn',s_train_knn)
    # print('poly',s_poly)
    # print('weight', s_weight)
    # print('ada+poly',s_ada_poly )
    s_ada = np.mean(s_ada)
    s_ada_knn= np.mean(s_ada_knn)
    s_knn= np.mean(s_knn)
    # s_ada = np.sqrt(mean_squared_error(s_ada,np.array([0]*len(s_ada))))
    # s_ada_knn= np.sqrt(mean_squared_error(s_ada_knn,np.array([0]*len(s_ada_knn))))
    # s_knn= np.sqrt(mean_squared_error(s_knn,np.array([0]*len(s_knn))))
    print('ada', s_ada)
    print('knn',s_knn)
    print('knn_ada',s_ada_knn)
    return ada_lng,ada_lat,knn_lng,knn_lat,ada_knn_lng,ada_knn_lat,s_ada,s_knn,s_ada_knn,s_rate
