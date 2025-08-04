import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import time
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

def data_op_main(predict_x,train_data,lte_id):
    predict_x = predict_x.T

    train_data = np.array(train_data).T #(25721,19)
    train_x = train_data[0:9, ].T
    train_y = train_data[9:,]
    train_y_lng = train_y[0,:]
    train_y_lat = train_y[1,:]
    start = time.time()
    ### knn
    knn.fit(train_x, train_y_lng)
    knn_lng = knn.predict(predict_x)
    knn.fit(train_x, train_y_lat)
    knn_lat = knn.predict(predict_x)
    ###
    ###
    (scaler_adaboost, bdt_lng, bdt_lat) = joblib.load('adaboost_model_应用集/' + lte_id + '.pkl')
    adaboost_x = predict_x
    adaboost_x = scaler_adaboost.transform(adaboost_x)
    ada_lng = bdt_lng.predict(adaboost_x)
    train_x_addFeature = ada_lng.reshape(adaboost_x.shape[0], 1)
    train_x_addFeature = np.hstack((adaboost_x, train_x_addFeature))
    ada_lat = bdt_lat.predict(train_x_addFeature)
    w = 0.8
    ada_knn_lng = w * ada_lng + (1-w) * knn_lng
    ada_knn_lat = w * ada_lat + (1-w) * knn_lat
 ###
    end = time.time()


    # print('s_train_knn',s_train_knn)
    # print('poly',s_poly)
    # print('weight', s_weight)
    # print('ada+poly',s_ada_poly )
    return ada_lng,ada_lat,knn_lng,knn_lat,ada_knn_lng,ada_knn_lat,end-start
