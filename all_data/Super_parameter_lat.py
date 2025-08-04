import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import joblib
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

def get_parameter_lat(data,BS_lng,BS_lat,limli_scale):
    # data = pd.read_excel('183956-8/183956-8.xlsx')
    data = np.array(data)  # (25721,19)
    data = data.T  # (19, 25721)
    x = data[0:9, :]  # (17, 25721)
    y = data[9:, ]  # (2, 25721)
    y_lat = y[1, :]
    y_lng = y[0, :]
    # BS_lng = 103.7025
    # BS_lat = 36.077222

    ## 除掉超过500米的
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
    ### 超参数
    min_error = 1
    for curr_parameter in range(1,5):
        curr_error = 0
        for i in range(7):
            train_x, test_x, train_y, test_y = train_test_split(x, y_lat, test_size=0.15,shuffle=True)
            ## 多项式回归
            poly_reg = PolynomialFeatures(degree=curr_parameter)
            X_Poly = poly_reg.fit_transform(train_x)
            lin_reg_2 = linear_model.LinearRegression()
            lin_reg_2.fit(X_Poly, train_y)
            train_y_pred = lin_reg_2.predict(poly_reg.fit_transform(train_x))
            test_y_pred = lin_reg_2.predict(poly_reg.fit_transform(test_x))
            ## 多项式回归结束
            #############计算误差
            # print(np.mean(np.abs(train_y-train_y_pred)))
            curr_error += (np.mean(np.abs(test_y-test_y_pred))/7.)
        if curr_error<min_error:
            parameter = curr_parameter
            min_error = curr_error
    # print(min_error)
    return parameter





