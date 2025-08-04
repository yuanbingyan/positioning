from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
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
k = 2
knn = KNeighborsRegressor(k)
def data_op_main(predict_x,train_data,lte_id):
    predict_x = predict_x.T
    predict_y = predict_x[9:,]
    predict_x = predict_x[0:9,].T  ## 准备需要预测的数据
    train_data = np.array(train_data).T #(25721,19)
    train_x = train_data[0:9,].T
    train_y = train_data[9:,]
    train_y_lng = train_y[0,:]
    train_y_lat = train_y[1,:]


    knn.fit(train_x, train_y_lng)
    pred_lng = knn.predict(predict_x)
    knn.fit(train_x, train_y_lat)
    pred_lat = knn.predict(predict_x)
    s = np.mean(distance(pred_lng, predict_y[0, :], pred_lat, predict_y[1, :]))
    print(s)
    return s

def pred():  # 预测数据
    limli_scale_data = pd.read_csv('计算覆盖距离.csv', encoding='ANSI')
    s_knn = []
    for index, row in limli_scale_data.iterrows():
        ## 准备训练数据
        lte_id = row['lte_id']
        data = np.load('data/{}.npz'.format(lte_id))
        lte_id_data = data['arr_0']
        # 准备预测数据
        predict_x = data['arr_1']
        s_knn.append(data_op_main(predict_x,lte_id_data,lte_id))
    print('mean:',np.mean(s_knn))
    print(np.sum(s_knn))
        ####

pred()