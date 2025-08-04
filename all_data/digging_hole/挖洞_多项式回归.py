import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from All_data.挖洞.Super_parameter_lat import get_parameter_lat
from All_data.挖洞.Super_parameter_lng import get_parameter_lng
from sklearn.externals import joblib
import os
import matplotlib.pyplot as plt
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
def polynomial_regression_lat(train_x,train_y,test_x,test_y,BS_lng,BS_lat,limli_scale,parameter_lng,lte_id):
    ## 多项式回归
    poly_reg = PolynomialFeatures(degree=parameter_lng)
    X_Poly = poly_reg.fit_transform(train_x)
    lin_reg_2 = linear_model.LinearRegression()
    lin_reg_2.fit(X_Poly, train_y)
    train_y_pred = lin_reg_2.predict(poly_reg.fit_transform(train_x))
    test_y_pred = lin_reg_2.predict(poly_reg.fit_transform(test_x))
    ## 多项式回归结束
    #############计算误差
    # print('纬度训练误差',np.mean(np.abs(train_y-train_y_pred)),len(train_y),len(test_y))
    # print('纬度测试误差',np.mean(np.abs(test_y-test_y_pred)))
    return test_y_pred
def polynomial_regression_lng(train_x,train_y,test_x,test_y,BS_lng,BS_lat,limli_scale,parameter_lng,lte_id):

    ## 多项式回归
    poly_reg = PolynomialFeatures(degree=parameter_lng)
    X_Poly = poly_reg.fit_transform(train_x)
    lin_reg_2 = linear_model.LinearRegression()
    lin_reg_2.fit(X_Poly, train_y)
    train_y_pred = lin_reg_2.predict(poly_reg.fit_transform(train_x))
    test_y_pred = lin_reg_2.predict(poly_reg.fit_transform(test_x))
    ## 多项式回归结束
    #############计算误差
    # print('经度训练误差',np.mean(np.abs(train_y-train_y_pred)),len(train_y),len(test_y))
    # print('经度测试误差',np.mean(np.abs(test_y-test_y_pred)))
    return test_y_pred
def train_poly(file,lng_start,lat_start): ## 训练模型
    limli_scale_data=pd.read_csv('计算覆盖距离.csv',encoding='ANSI')
    data_with_GPS = pd.read_excel(file)

    lte_id = file.split('.')[0]
    limli_scale = 500
    BS_lng = data_with_GPS[data_with_GPS['lte_id'] == lte_id].经度.iloc[0]
    BS_lat = data_with_GPS[data_with_GPS['lte_id'] == lte_id].维度.iloc[0]
    lte_id_data = data_with_GPS[data_with_GPS['lte_id'] == lte_id][['ltescrsrp','ltescrsrq','ltescphr','ltencrsrp_1','ltencrsrq_1','ltencpci_1','ltencrsrp_2','ltencrsrq_2','ltencpci_2','longitude','latitude']]
    # 获取超参数
    parameter_lat = get_parameter_lat(lte_id_data,BS_lng,BS_lat,limli_scale)
    parameter_lng = get_parameter_lng(lte_id_data,BS_lng,BS_lat,limli_scale)
    ## data
    data = np.array(lte_id_data)  # (25721,19)
    data = data.T  # (19, 25721)
    x = data[0:9, :]  # (17, 25721)
    y = data[9:, ]  # (2, 25721)
    y_lat = y[1, :]
    y_lng = y[0, :]
    s = distance(BS_lng, y_lng, BS_lat, y_lat)
    num = []
    for j in range(len(s)):
        if s[j] > limli_scale:
            num.append(j)
    # print('500米范围占',1-len(num)/len(s))
    x = np.delete(x, num, axis=1)
    y = np.delete(y, num, axis=1)
    y_lng = y[0, :]
    y_lat = y[1, :]
    ############ 数据处理
    x = x.T
    scaler = preprocessing.MinMaxScaler()
    x = scaler.fit_transform(x)
    y = y.T
    ###
    num_test = []
    num_train = []
    for i in range(len(y_lng)):
        if (lat_start<y_lat[i]<lat_start+0.0009) and (lng_start<y_lng[i]<lng_start+0.0011):
           num_test.append(i)
        else:num_train.append(i)
    train_x = x[num_train]
    train_y = y[num_train]
    test_x = x[num_test]
    test_y = y[num_test]

    ###
    # 多项式回归法训练
    lng_pred = polynomial_regression_lng(train_x,train_y[:,0],test_x,test_y[:,0],BS_lng,BS_lat,limli_scale,parameter_lng,lte_id)
    lat_pred = polynomial_regression_lat(train_x,train_y[:,1],test_x,test_y[:,1],BS_lng,BS_lat,limli_scale,parameter_lat,lte_id)
    # print(lng_pred.shape,lat_pred.shape,train_y.shape,test_y.shape)
    # print(lte_id)
    i = 0
    for j in range(len(lng_pred)):
        if lng_start<lng_pred[j]<lng_start+0.0011 and lat_start<lat_pred[j]<lat_start+0.0009:
            i+=1
    print('poly:','train:',train_y.shape,'test:',test_y.shape,'digit:',i)
    plt.scatter(train_y[:,0],train_y[:,1],s=0.5,c='r')
    plt.scatter(test_y[:,0],test_y[:,1],s=0.5,c='b')
    plt.scatter(lng_pred, lat_pred, s=0.5, c='g')
    plt.show()
# train_poly('183053-1.xlsx',103.704,36.074)