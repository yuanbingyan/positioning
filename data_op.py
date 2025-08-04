import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
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

def data_op_main(predict_x,train_data,lte_id):

    predict_x = np.array(predict_x)  ## 准备需要预测的数据
    ## 纬度预测
    knn_lng,knn_lat = joblib.load('knn_model/'+str(lte_id)+'.pkl')
    knn_lat_pred = knn_lat.predict(predict_x)
    ## 经度预测
    knn_lng_pred = knn_lng.predict(predict_x)

    (scaler_adaboost, bdt_lng, bdt_lat) = joblib.load('adaboost_model/'+str(lte_id)+'.pkl')
    adaboost_x = predict_x
    adaboost_x = scaler_adaboost.transform(adaboost_x)
    ada_lng_pred = bdt_lng.predict(adaboost_x)
    train_x_addFeature = ada_lng_pred.reshape(adaboost_x.shape[0],1)
    train_x_addFeature = np.hstack((adaboost_x,train_x_addFeature))
    ada_lat_pred = bdt_lat.predict(train_x_addFeature)

    #加权
    w = 0.8
    ada_knn_lng_pred = w * ada_lng_pred + (1 - w) * knn_lng_pred
    ada_knn_lat_pred = w * ada_lat_pred + (1 - w) * knn_lat_pred

    return ada_knn_lng_pred,ada_knn_lat_pred
