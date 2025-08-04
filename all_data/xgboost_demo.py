from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

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

xlf = xgb.XGBRegressor( max_depth=9,
                        learning_rate=0.1,
                        n_estimators=150,
                        min_child_weight=1,
                        subsample=0.8,
                        reg_alpha=0,
                        reg_lambda=1,
            gamma=0,)
def data_op_main(predict_x,train_data,lte_id):
    predict_x = predict_x.T
    predict_y = predict_x[9:,]
    predict_x = predict_x[0:9,].T  ## 准备需要预测的数据
    train_data = np.array(train_data).T #(25721,19)
    train_x = train_data[0:9,].T
    train_y = train_data[9:,]
    train_y_lng = train_y[0,:]
    train_y_lat = train_y[1,:]


    xlf.fit(train_x, train_y_lng)
    pred_lng = xlf.predict(predict_x)
    xlf.fit(train_x, train_y_lat)
    pred_lat = xlf.predict(predict_x)
    s = distance(pred_lng, predict_y[0, :], pred_lat, predict_y[1, :])
    s_xgboost = np.sqrt(mean_squared_error(s, np.array([0] * len(s))))
    print(s_xgboost)
    return s_xgboost

def pred():  # 预测数据
    limli_scale_data = pd.read_csv('计算覆盖距离.csv', encoding='ANSI')
    s_xgboost = []
    for index, row in limli_scale_data.iterrows():
        ## 准备训练数据
        lte_id = row['lte_id']
        data = np.load('data/{}.npz'.format(lte_id))
        lte_id_data = data['arr_0']
        # 准备预测数据
        predict_x = data['arr_1']
        s_xgboost.append(data_op_main(predict_x,lte_id_data,lte_id))
    print('mean:',np.mean(s_xgboost))
    print(np.sum(s_xgboost))
        ####

pred()